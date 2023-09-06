import logging
import os
import sys
import json
import yaml
#
import configargparse as argparse
import torch
import numpy as np
import re

import gunpowder as gp
import incasem as fos

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.getLogger('gunpowder').setLevel(logging.INFO)

class RunDummy():
    def __init__(self):
        # # Get the highest ID and add 1
        # pred_jsons = [int(e.split(".json")[0]) for e in os.listdir("json_configs")]
        # train_jsons = [int(e.split(".json")[0]) for e in os.listdir("../02_train/json_configs")]
        # 
        # ids = pred_jsons + train_jsons
        # self._id = max(-1, max(ids))+1
        self._id=0

        self.log = []

    def log_scalar(self, name, value, step):
        self.log.append({"name": name, "value": value, "step": step})

def torch_setup(_config):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def model_setup(_run_dummy, _config):
    model_type = _config['model']['type']
    if model_type == 'OneConv3d':
        model = fos.torch.models.OneConv3d(
            out_channels=_config['model']['num_fmaps_out']
        )
    elif model_type == 'Unet':
        model = fos.torch.models.Unet(
            in_channels=1,
            num_fmaps=int(_config['model']['num_fmaps']),
            fmap_inc_factor=int(_config['model']['fmap_inc_factor']),
            downsample_factors=tuple(
                tuple(i) for i in _config['model']['downsample_factors']
            ),
            # kernel_size_down=None,
            # kernel_size_up=None,
            # activation='ReLU',
            voxel_size=_config['data']['voxel_size'],
            num_fmaps_out=_config['model']['num_fmaps_out'],
            # num_heads=1,
            constant_upsample=_config['model']['constant_upsample'],
            padding='valid'
        )
    elif model_type == 'MultitaskUnet':
        model = fos.torch.models.MultitaskUnet(
            _config['model']['num_fmaps_out'],
            _config['model']['num_fmaps_out_auxiliary'],
            dims=3,
            in_channels=1,
            num_fmaps=int(_config['model']['num_fmaps']),
            fmap_inc_factor=int(_config['model']['fmap_inc_factor']),
            downsample_factors=tuple(
                tuple(i) for i in _config['model']['downsample_factors']
            ),
            # kernel_size_down=None,
            # kernel_size_up=None,
            # activation='ReLU',
            voxel_size=_config['data']['voxel_size'],
            # num_heads=1,
            constant_upsample=_config['model']['constant_upsample'],
            padding='valid'
        )
    else:
        raise ValueError(f"Model type {model_type} does not exist.")

    model.eval()

    total_params = sum(p.numel()
                       for p in model.parameters())
    logger.info(f'{total_params=}')
    _run_dummy.log_scalar('num_params', total_params, 0)

    return model


def directory_structure_setup(_run_dummy, _config):

    predictions_out_path = os.path.expanduser(
        _config['prediction']['directories']['prefix'])
    if not os.path.isdir(predictions_out_path):
        os.makedirs(predictions_out_path)

    # training run id, then prediction run id as subfolder

    run_path = os.path.join(
        f"{int(_config['prediction']['run_id_training']):04d}",
        _config["prediction"]["name"]
    )
    return run_path


def get_checkpoint(checkpoint_file):
    # TODO checkpoint relative to run path
    # TODO get latest checkpoint of the specified run automatically
    logger.debug(f"{checkpoint_file=}")
    logger.debug(f"{type(checkpoint_file)=}")
    if checkpoint_file is None:
        raise ValueError(
            f"Specify a checkpoint for making predictions")

    return checkpoint_file


def multiple_prediction_setup(_config, run_path, model_, checkpoint):
    prediction_datasets = fos.utils.create_multiple_config(
        _config['prediction']['data'])
    pred_setups = []
    for pred_ds in prediction_datasets:
        pred_setups.append(
            prediction_setup(
                _config,
                run_path,
                model_,
                checkpoint,
                pred_ds)
        )
    return pred_setups


def prediction_setup(_config, run_path, model_,
                     checkpoint, pred_dataset):
    pipeline_type = {
        'baseline': fos.pipeline.PredictionBaseline,
    }[_config['prediction']['pipeline']]

    prediction = pipeline_type(
        data_config=pred_dataset,
        run_id=run_path,
        data_path_prefix=os.path.expanduser(_config['directories']['data']),
        predictions_path_prefix=os.path.expanduser(
            _config['prediction']['directories']['prefix']),
        model=model_,
        num_classes=int(_config['data']['num_classes']),
        voxel_size=_config['data']['voxel_size'],
        input_size_voxels=_config['prediction']['input_size_voxels'],
        output_size_voxels=_config['prediction']['output_size_voxels'],
        checkpoint=checkpoint,
    )
    prediction.predict.gpus = [int(_config['prediction']['torch']['device'])]
    prediction.scan.num_workers = _config['prediction']['num_workers']

    return prediction


def remove_context(batch, input_size_voxels, output_size_voxels):
    voxel_size = batch[gp.ArrayKey('RAW')].spec.voxel_size
    roi = batch[gp.ArrayKey('RAW')].spec.roi
    context = (
        gp.Coordinate(input_size_voxels) -
        gp.Coordinate(output_size_voxels)

    ) / 2
    context = context * voxel_size
    roi = roi.grow(-context, -context)

    for key, array in batch.arrays.items():
        batch[key] = array.crop(roi)

    return batch


def log_metrics(
        _run_dummy,
        target,
        prediction_probas,
        mask,
        metric_mask,
        run_path,
        iteration,
        mode):
    mask = np.logical_and(mask.astype(bool), metric_mask.astype(bool))

    jaccard_scores = []
    for i in range(prediction_probas.shape[0]):
        jac_score = fos.metrics.pairwise_distance_metric_thresholded(
            target=target,
            prediction_probas=prediction_probas,
            metric='jaccard',
            threshold=0.5,
            foreground_class=i,
            mask=mask
        )
        jaccard_scores.append(jac_score)
    for label, score in enumerate(jaccard_scores):
        _run_dummy.log_scalar(f"jaccard_class_{label}_{mode}", score, iteration)
        logger.info(f"{mode} | Jaccard score class {label}: {score}")

    dice_scores = []
    for i in range(prediction_probas.shape[0]):
        dic_score = fos.metrics.pairwise_distance_metric_thresholded(
            target=target,
            prediction_probas=prediction_probas,
            metric='dice',
            threshold=0.5,
            foreground_class=i,
            mask=mask
        )
        dice_scores.append(dic_score)
    for label, score in enumerate(dice_scores):
        _run_dummy.log_scalar(f"dice_class_{label}_{mode}", score, iteration)
        logger.info(f"{mode} | Dice score class {label}: {score}")


    precision_recall = fos.metrics.precision_recall(
        target,
        prediction_probas,
        mask
    )
    for i, (p, r) in enumerate(precision_recall):
        _run_dummy.log_scalar(f"precision_{i}_{mode}", p, iteration)
        logger.info(
            f"{mode} | Precision class {i}: {p}")
        _run_dummy.log_scalar(f"recall_{i}_{mode}", r, iteration)
        logger.info(
            f"{mode} | Recall class {i}: {r}")


def predict(_run_dummy, _config, checkpoint=None, iteration=0, run_path=None):
    """predict.

    """

    torch_setup(_config)

    if run_path is None:
        run_path = directory_structure_setup(_run_dummy,_config)
    model = model_setup(_run_dummy,_config)

    if checkpoint is None:
        checkpoint = get_checkpoint(_config['prediction']['checkpoint'])

    predictions = multiple_prediction_setup(
        _config, run_path=run_path, model_=model, checkpoint=checkpoint)

    for idx_pipeline, prediction in enumerate(predictions):
        with gp.build(prediction.pipeline) as p:
            request = gp.BatchRequest()

            if _config['prediction']['log_metrics']:
                provider_spec = p.spec
                for key, spec in provider_spec.items():
                    if key in prediction.request:
                        request_spec = spec.copy()
                        request_spec.dtype = None
                        request[key] = request_spec

            # labels_roi = request[gp.ArrayKey('LABELS')].roi
            # predictions_roi = request[gp.ArrayKey('PREDICTIONS')].roi
            # assert labels_roi == predictions_roi, \
            # (f"{labels_roi=} and {predictions_roi=} do not match, "
            # "probably due to some padding of the dataset "
            # "while building the pipeline")

            # logger.debug(f"Total request: {request}")

            batch = p.request_batch(request)

            # # explicitly remove context, even though it is already removed
            # # as a side effect of the metric mask request at output_size
            # batch = remove_context(
            # batch,
            # _config['prediction']['input_size_voxels'],
            # _config['prediction']['output_size_voxels']
            # )

            # TODO load files from disk as daisy datasets
            if _config['prediction']['log_metrics']:
                log_metrics(
                    _run_dummy,
                    target=batch[gp.ArrayKey('LABELS')].data,
                    prediction_probas=batch[gp.ArrayKey('PREDICTIONS')].data,
                    mask=batch[gp.ArrayKey('MASK')].data,
                    metric_mask=batch[gp.ArrayKey('METRIC_MASK')].data,
                    run_path=run_path,
                    iteration=iteration,
                    mode=f'ds_{idx_pipeline}'
                )


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--run_id',
        '-r',
        type=int,
        required=True,
        help='Run ID of training that is used to make predictions'
    )

    args, remaining_argv = parser.parse_known_args()

    remaining_argv_dict = {}

    # Extra parsing
    if "--name"in remaining_argv:
        name_idx = remaining_argv.index("--name") + 1
        name = remaining_argv[name_idx]
        remaining_argv_dict['name'] = name

    if "with" in remaining_argv:
        with_idx = remaining_argv.index("with")
        cfg_yaml_path = remaining_argv[with_idx+1]
        remaining_argv_dict["cfg_yaml"] = cfg_yaml_path

    for item in remaining_argv:
        if "prediction." in item and "=" in item:
            pattern = r'\bprediction\.(\S+)\s*=\s*(\S+)\b'
            # Find all matches in the text
            matches = re.findall(pattern, item)
            if len(matches)>0:
                k, v = item.split("prediction.")[-1].split("=")
                remaining_argv_dict[k] = v

    return args, remaining_argv_dict


if __name__ == '__main__':

    args, remaining_argv = parse_arguments()

    available_models = [int(e.split(".json")[0]) for e in os.listdir("json_configs")]

    assert args.run_id in available_models, "Desired run_id not found in json_configs, make sure it exists"

    json_file = f"json_configs/{args.run_id}.json"

    with open(json_file) as f:
        config = json.load(f)

    # Read the YAML file into a dictionary
    yaml_data = {}
    if "cfg_yaml" in remaining_argv:
        with open(remaining_argv['cfg_yaml'], 'r') as file:
            yaml_data = yaml.safe_load(file)

    # Add remaining_argv and config yaml to config
    config = {**config, **yaml_data}
    config["prediction"] = {**config["prediction"], **remaining_argv}
    config["prediction"]["run_id_training"] = args.run_id

    _run_dummy = RunDummy()

    predict(_run_dummy, config)
