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

SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))


def resolve_repo_path(path):
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(REPO_ROOT, expanded))


class PredictionRunDummy():
    def __init__(self):
        # Get the highest ID and add 1
        with open(resolve_repo_path('mock_db/ledger.json')) as f:
            ledger = json.load(f)

        ids = [int(e) for e in ledger.keys()]

        self._id = max(-1, max(ids)) + 1

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
    predictions_out_path = resolve_repo_path(
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
        data_path_prefix=resolve_repo_path(_config['directories']['data']),
        predictions_path_prefix=resolve_repo_path(
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
        run_path = directory_structure_setup(_run_dummy, _config)
    model = model_setup(_run_dummy, _config)

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
    return run_path


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
    if "--name" in remaining_argv:
        name_idx = remaining_argv.index("--name") + 1
        name = remaining_argv[name_idx]
        remaining_argv_dict['name'] = name

    if "with" in remaining_argv:
        with_idx = remaining_argv.index("with")
        cfg_yaml_path = remaining_argv[with_idx + 1]
        remaining_argv_dict["cfg_yaml"] = cfg_yaml_path

    for item in remaining_argv:
        if "prediction." in item and "=" in item:
            pattern = r'\bprediction\.(\S+)\s*=\s*(\S+)\b'
            # Find all matches in the text
            matches = re.findall(pattern, item)
            if len(matches) > 0:
                k, v = item.split("prediction.")[-1].split("=")
                remaining_argv_dict[k] = v

    return args, remaining_argv_dict


def _snap_to_mod(value, mod, target):
    """Return nearest integer to value that is congruent to target (mod)."""
    value = int(round(value))
    if mod <= 0:
        return value
    delta = (target - (value % mod)) % mod
    up = value + delta
    down = value - ((mod - delta) % mod)
    if down <= 0:
        return up
    # Prefer the closer one; tie-break upwards to avoid tiny ROIs.
    if (value - down) <= (up - value):
        return down
    return up


def _apply_resolution_agnostic_pred_sizes(config):
    """
    Compute prediction output/input sizes from a fixed physical output size.

    Assumes the current U-Net architecture with output = input - 94 voxels.
    """
    physical_output_nm = (550, 550, 550)
    voxel_size = config['data']['voxel_size']
    output_voxels = []
    for phys, vs in zip(physical_output_nm, voxel_size):
        output_raw = phys / vs
        # Output dims must be 6 (mod 8) so input dims end up 4 (mod 8).
        output_voxels.append(_snap_to_mod(output_raw, 8, 6))

    output_voxels = [int(v) for v in output_voxels]
    input_voxels = [int(v + 94) for v in output_voxels]

    config['prediction']['output_size_voxels'] = output_voxels
    config['prediction']['input_size_voxels'] = input_voxels


def _apply_data_voxel_size_from_prediction_sources(config):
    """Derive data.voxel_size from the prediction data config JSON."""
    data_config_path = config['prediction'].get('data')
    if not data_config_path:
        raise ValueError("prediction.data is required to infer voxel_size.")
    data_config_path = resolve_repo_path(data_config_path)
    with open(data_config_path, 'r') as f:
        data_sources = json.load(f)
    if not data_sources:
        raise ValueError(f"No datasets found in {data_config_path}.")
    first = next(iter(data_sources.values()))
    try:
        voxel_size = first['voxel_size']
    except KeyError as e:
        raise ValueError(
            f"voxel_size missing in prediction data config {data_config_path}."
        ) from e
    config['data']['voxel_size'] = voxel_size


if __name__ == '__main__':

    args, remaining_argv = parse_arguments()
    with open(resolve_repo_path('mock_db/ledger.json')) as fp:
        ledger = json.load(fp)
    available_models = [int(e) for e in ledger.keys()]

    assert args.run_id in available_models, "Desired run_id not found in mock_db, make sure it exists"

    json_file = resolve_repo_path(f"mock_db/{ledger[str(args.run_id)]}")

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
    _apply_data_voxel_size_from_prediction_sources(config)
    _apply_resolution_agnostic_pred_sizes(config)

    _run_dummy = PredictionRunDummy()

    try:
        name = config["prediction"]["name"]
    except KeyError:
        name = "prediction"

    name = name + f"_{_run_dummy._id}"

    # Update ledger
    ledger[str(_run_dummy._id)] = name

    with open(resolve_repo_path("mock_db/ledger.json"), mode="w") as f:
        json.dump(ledger, f)

    with open(resolve_repo_path(config["prediction"]["data"])) as f:
        prediction_data = json.load(f)

    prediction_data_file = [e for e in prediction_data.keys()][0]
    prediction_data_filename = prediction_data[prediction_data_file]["file"]
    prediction_data_path = os.path.join(
        resolve_repo_path(config["directories"]["data"]),
        prediction_data_filename
    )

    results_path = predict(_run_dummy, config)
    data_path = config["directories"]["data"]

    logger.info(
        "Prediction named: {} was written to {}/predictions/{}".format(name, prediction_data_path, results_path))
