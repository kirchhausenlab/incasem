import os
from shutil import copyfile
import sys
import logging
import argparse
import json
import re
import yaml
from pathlib import Path

import numpy as np
import torch
import tensorboardX

import gunpowder as gp

import incasem as fos


class TrainingRunDummy:
    def __init__(self):
        # Get the highest ID and add 1
        ledger_path = (
            Path(__file__).resolve().parents[2].joinpath("mock_db/ledger.json")
        )

        with open(f"{ledger_path}") as f:
            ledger = json.load(f)

        ids = [int(e) for e in ledger.keys()]

        self._id = max(-1, max(ids)) + 1

        self.log = []

    def log_scalar(self, name, value, step):
        self.log.append({"name": name, "value": value, "step": step})


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("gunpowder").setLevel(logging.INFO)


def torch_setup(_config):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if _config["torch"]["device"] == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def model_setup(_config, _run_dummy):
    model_type = _config["model"]["type"]
    if model_type == "OneConv3d":
        model = fos.torch.models.OneConv3d(
            out_channels=_config["model"]["num_fmaps_out"]
        )
    elif model_type == "Unet":
        model = fos.torch.models.Unet(
            in_channels=1,
            num_fmaps=int(_config["model"]["num_fmaps"]),
            fmap_inc_factor=int(_config["model"]["fmap_inc_factor"]),
            downsample_factors=tuple(
                tuple(i) for i in _config["model"]["downsample_factors"]
            ),
            activation="ReLU",
            voxel_size=_config["data"]["voxel_size"],
            num_fmaps_out=_config["model"]["num_fmaps_out"],
            num_heads=1,
            constant_upsample=_config["model"]["constant_upsample"],
            padding="valid",
        )
    elif model_type == "MultitaskUnet":
        model = fos.torch.models.MultitaskUnet(
            _config["model"]["num_fmaps_out"],
            _config["model"]["num_fmaps_out_auxiliary"],
            dims=3,
            in_channels=1,
            num_fmaps=int(_config["model"]["num_fmaps"]),
            fmap_inc_factor=int(_config["model"]["fmap_inc_factor"]),
            downsample_factors=tuple(
                tuple(i) for i in _config["model"]["downsample_factors"]
            ),
            activation="ReLU",
            voxel_size=_config["data"]["voxel_size"],
            constant_upsample=_config["model"]["constant_upsample"],
            padding="valid",
        )
    else:
        raise ValueError(f"Model type {model_type} does not exist.")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"{total_params=}")

    _run_dummy.log_scalar("num_params", total_params, 0)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{trainable_params=}")
    return model


def loss_setup(_config, device="cuda"):
    weight = torch.tensor(list(_config["loss"]["weight"]), dtype=torch.float)

    loss_type = _config["loss"]["type"]
    if loss_type == "cross_entropy_scaling":
        loss = fos.torch.loss.CrossEntropyLossWithScalingAndMeanReduction(
            weight=weight, device=device
        )
    # elif loss_type == 'cross_entropy':
    # loss = torch.nn.CrossEntropyLoss(weight=weight, device=device)
    else:
        raise ValueError(f"Specified loss {loss_type} does not exist.")

    return loss


def training_setup(_config, _run_dummy, _seed, run_dir, model):
    loss = loss_setup(_config)

    # TODO parametrize type of optimizer, move to separate function
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(_config["training"]["optimizer"]["lr"]),
        weight_decay=float(_config["training"]["optimizer"]["weight_decay"]),
    )

    pipeline_type = {"baseline_with_context": fos.pipeline.TrainingBaselineWithContext}[
        _config["training"]["pipeline"]
    ]

    training = pipeline_type(
        data_config=_config["training"]["data"],
        run_dir=run_dir,
        run_path_prefix=os.path.expanduser(_config["directories"]["runs"]),
        data_path_prefix=os.path.expanduser(_config["directories"]["data"]),
        model=model,
        loss=loss,
        optimizer=optimizer,
        num_classes=int(_config["data"]["num_classes"]),
        voxel_size=_config["data"]["voxel_size"],
        input_size_voxels=_config["training"]["input_size_voxels"],
        output_size_voxels=_config["training"]["output_size_voxels"],
        reject_min_masked=float(_config["training"]["reject"]["min_masked"]),
        reject_probability=float(_config["training"]["reject"]["reject_probability"]),
        random_seed=_seed,
    )

    device = _config["torch"]["device"]
    training.train_node.gpus = [] if device == "cpu" else [int(device)]

    training.train_node.save_every = int(_config["training"]["save_every"])
    training.train_node.log_every = int(_config["training"]["log_every"])

    # Downsample
    training.downsample.factor = int(_config["data"]["downsample_factor"])

    # Balance Labels
    try:
        training.balance_labels.clipmin = float(
            _config["loss"]["balance_labels"]["clipmin"]
        )
        training.balance_labels.clipmax = float(
            _config["loss"]["balance_labels"]["clipmax"]
        )
    except AttributeError:
        logger.warning(f"Trying to set BalanceLabels attributes, but it is not used.")
        _config["loss"]["balance_labels"]["clipmin"] = None
        _config["loss"]["balance_labels"]["clipmax"] = None

    try:
        training.augmentation.nodes["simple_0"].transpose_only = _config["training"][
            "augmentation"
        ]["simple"]["transpose_only"]
    except (KeyError, AttributeError):
        logger.warning("SimpleAugment 0 transpose only not set.")

    try:
        training.augmentation.nodes["elastic"].control_point_spacing = tuple(
            _config["training"]["augmentation"]["elastic"]["control_point_spacing"]
        )
        training.augmentation.nodes["elastic"].jitter_sigma = tuple(
            _config["training"]["augmentation"]["elastic"]["jitter_sigma"]
        )
        training.augmentation.nodes["elastic"].subsample = int(
            _config["training"]["augmentation"]["elastic"]["subsample"]
        )
    except (KeyError, AttributeError):
        logger.warning(
            "Trying to set parameters for ElasticAugment, but it is not used."
        )

    try:
        training.augmentation.nodes["simple_1"].transpose_only = _config["training"][
            "augmentation"
        ]["simple"]["transpose_only"]
    except (KeyError, AttributeError):
        logger.warning("SimpleAugment 1 transpose only not set.")

    try:
        training.augmentation.nodes["intensity"].scale_min = 1.0 - float(
            _config["training"]["augmentation"]["intensity"]["scale"]
        )
        training.augmentation.nodes["intensity"].scale_max = 1.0 + float(
            _config["training"]["augmentation"]["intensity"]["scale"]
        )
        training.augmentation.nodes["intensity"].shift_min = -1.0 * float(
            _config["training"]["augmentation"]["intensity"]["shift"]
        )
        training.augmentation.nodes["intensity"].shift_max = float(
            _config["training"]["augmentation"]["intensity"]["shift"]
        )
    except (KeyError, AttributeError):
        logger.warning(
            "Trying to set parameters for IntensityAugment, but it is not used."
        )

    # Precache
    try:
        training.precache.cache_size = int(
            _config["training"]["precache"]["cache_size"]
        )
        training.precache.num_workers = int(
            _config["training"]["precache"]["num_workers"]
        )
    except AttributeError:
        logger.warning(f"Trying to set Precache attributes, but it is not used.")
    except KeyError:
        logger.warning(
            f"Trying to set Precache attributes, but not specified in config."
        )

    # Snapshot
    training.snapshot.every = int(_config["training"]["snapshot"]["every"])

    # Profiling Stats
    training.profiling_stats.every = int(
        _config["training"]["profiling_stats"]["every"]
    )

    return training


def multiple_validation_setup(_config, _run_dummy, _seed, run_dir, model):
    val_datasets = fos.utils.create_multiple_config(_config["validation"]["data"])

    validations = []
    for val_ds in val_datasets:
        validations.append(
            validation_setup(
                _config,
                _run_dummy,
                _seed,
                run_dir,
                model,
                val_ds,
            )
        )
    return validations


def validation_setup(_config, _run_dummy, _seed, run_dir, model, val_dataset):
    # Validation loss is assumed to be the same as the training loss
    loss = loss_setup(_config)

    pipeline_type = {
        "baseline_with_context": fos.pipeline.ValidationBaselineWithContext
    }[_config["validation"]["pipeline"]]

    validation = pipeline_type(
        data_config=val_dataset,
        run_dir=run_dir,
        run_path_prefix=os.path.expanduser(_config["directories"]["runs"]),
        data_path_prefix=os.path.expanduser(_config["directories"]["data"]),
        model=model,
        loss=loss,
        num_classes=int(_config["data"]["num_classes"]),
        voxel_size=_config["data"]["voxel_size"],
        input_size_voxels=_config["validation"]["input_size_voxels"],
        output_size_voxels=_config["validation"]["output_size_voxels"],
        run_every=_config["validation"]["validate_every"],
        random_seed=_seed,
    )
    device = _config["torch"]["device"]
    validation.predict.gpus = [] if device == "cpu" else [int(device)]

    # Downsample
    validation.downsample.factor = int(_config["data"]["downsample_factor"])

    # Balance Labels
    try:
        validation.balance_labels.clipmin = float(
            _config["loss"]["balance_labels"]["clipmin"]
        )
        validation.balance_labels.clipmax = float(
            _config["loss"]["balance_labels"]["clipmax"]
        )
    except AttributeError:
        logger.warning(f"Trying to set BalanceLabels attributes, but it is not used.")
        _config["loss"]["balance_labels"]["clipmin"] = None
        _config["loss"]["balance_labels"]["clipmax"] = None

    validation.snapshot.every = int(_config["validation"]["snapshot"]["every"])

    return validation


def log_result(
    _run_dummy,
    _config,
    metric_name="loss",
    metric_val=float("inf"),
):
    # TODO This should be some metric, not the loss function
    try:
        _config["name"]
    except:
        _config["name"] = "training_{}".format(_run_dummy._id)

    experiment_name = "Run {}: {}".format(_run_dummy._id, _config["name"])

    return f"\n{experiment_name}" f"\nval {metric_name}: {metric_val:.6f}"


# def log_data_config(_config, _run_dummy):
#     _run_dummy.add_artifact(_config['training']['data'])

#     val_config_files = _config['validation']['data'].split(',')
#     for f in val_config_files:
#         _run_dummy.add_artifact(f)


def directory_structure_setup(_config, _run_dummy):
    try:
        dir_run_id = _config["training"]["continue_id"]
        logger.info(f"Continue training run {dir_run_id}")
    except KeyError:
        try:
            load_run_id, load_run_checkpoint = _config["training"]["start_from"]
            model_to_load = os.path.expanduser(load_run_checkpoint)
            # model_to_load = os.path.join(
            # os.path.expanduser(_config['directories']['runs']),
            # "models",
            # str(load_run_id),
            # "model_checkpoint_" + str(load_run_checkpoint))
            new_model = os.path.join(
                os.path.expanduser(_config["directories"]["runs"]),
                "models",
                str(_run_dummy._id),
                "model_checkpoint_0",
            )

            os.makedirs(os.path.dirname(new_model), exist_ok=True)
            copyfile(model_to_load, new_model)
            dir_run_id = _run_dummy._id

            logger.info(
                f"Starting new training run {dir_run_id}, \
                from previous run {load_run_id}, \
                checkpoint {load_run_checkpoint}"
            )
        except KeyError:
            dir_run_id = _run_dummy._id
            logger.info(f"Starting new training run {dir_run_id}")

    dir_run_id = f"{dir_run_id:04d}"
    return dir_run_id


def log_metrics(_run, target, prediction_probas, mask, metric_mask, iteration, mode):
    mask = np.logical_and(mask.astype(bool), metric_mask.astype(bool))

    dice_scores = []
    for i in range(prediction_probas.shape[0]):
        dic_score = fos.metrics.pairwise_distance_metric_thresholded(
            target=target,
            prediction_probas=prediction_probas,
            threshold=0.5,
            metric="dice",
            foreground_class=i,
            mask=mask,
        )
        dice_scores.append(dic_score)
    for label, score in enumerate(dice_scores):
        _run_dummy.log_scalar(f"dice_class_{label}_{mode}", score, iteration)
        logger.info(f"{mode} | Dice score class {label}: {score}")

    # jaccard_scores = fos.metrics.jaccard(
    # target,
    # prediction_probas,
    # mask
    # )
    # for label, score in enumerate(jaccard_scores):
    # _run_dummy.log_scalar(f"jaccard_class_{label}_{mode}", score, iteration)

    # average_precision_scores = fos.metrics.average_precision(
    # target,
    # prediction_probas
    # )
    # for label, score in enumerate(average_precision_scores):
    # _run_dummy.log_scalar(f"AP_class_{label}_{mode}", score, iteration)


def log_labels_balance(_run, labels, num_classes, iteration):
    try:
        for c in range(num_classes):
            pct = np.sum(labels == c) / np.prod(labels.shape)
            _run_dummy.log_scalar(f"pct_class_{c}", pct, iteration)
    except KeyError as e:
        logger.error(e)


def log_tb_batch_position(summary_writer, i, raw_pos):
    logger.debug(f"{i=}, {raw_pos=}")
    summary_writer.add_scalar("offset_z", int(raw_pos[0][0]), i)
    summary_writer.add_scalar("offset_y", int(raw_pos[0][1]), i)
    summary_writer.add_scalar("offset_x", int(raw_pos[0][2]), i)
    summary_writer.add_scalar("shape_z", int(raw_pos[1][0]), i)
    summary_writer.add_scalar("shape_y", int(raw_pos[1][1]), i)
    summary_writer.add_scalar("shape_x", int(raw_pos[1][2]), i)


def log_tb_batch_labels_balance(summary_writer, i, labels, num_classes):
    for c in range(num_classes):
        pct = np.sum(labels == c) / np.prod(labels.shape)
        summary_writer.add_scalar(f"pct_class_{c}", pct, i)


def train(_config, _run, _seed):
    """train.

    Args:
        data_config:
        val_data_config:
    """

    torch_setup(_config)
    # log_data_config(_config, _run)

    run_dir = directory_structure_setup(_config, _run)

    model = model_setup(_config, _run)
    training = training_setup(_config, _run, _seed, run_dir=run_dir, model=model)

    validations = multiple_validation_setup(
        _config, _run, _seed, run_dir=run_dir, model=model
    )
    validation_loss = float("inf")

    debug_logdir = os.path.join(
        os.path.expanduser(_config["directories"]["runs"]),
        "tensorboard",
        run_dir,
        "debug",
    )
    logger.info(f"{debug_logdir=}")
    debug_writer = tensorboardX.SummaryWriter(debug_logdir)

    # ### START ITERATING ### #
    with gp.build(training.pipeline) as p:
        # Hack for validation in continued training
        logger.info(
            (
                f"Training iteration is {training.train_node.iteration}, "
                "copying into validation pipeline"
            )
        )
        start_iteration = training.train_node.iteration

        # build validation pipelines
        for idx_pipeline, validation in enumerate(validations):
            try:
                validations[idx_pipeline].pipeline.setup()
            except BaseException:
                logger.error(
                    f"something went wrong during the setup of pipeline {idx_pipeline}, calling tear down"
                )
                validations[idx_pipeline].pipeline.internal_teardown()
                logger.debug("tear down completed")
                raise

            validations[idx_pipeline].validation_loss.iteration = start_iteration

        # from 0 to iterations+1, for logging once more in the end.
        for i in range(start_iteration, _config["training"]["iterations"] + 1):
            batch = p.request_batch(training.request)
            # logger.debug(f'batch {i}:\n{batch}')

            log_tb_batch_position(debug_writer, i, batch[gp.ArrayKey("RAW_POS")].data)
            log_tb_batch_labels_balance(
                debug_writer,
                i,
                batch[gp.ArrayKey("LABELS")].data,
                _config["data"]["num_classes"],
            )

            if i % _config["sacred"]["log_every"] == 0:
                # Convention for loss: pos 0 is the final loss used for
                # backprop, other positions are intermediate/partial losses
                for l_i, l in enumerate(np.atleast_1d(batch.loss)):
                    _run_dummy.log_scalar(f"loss_train_{l_i}", l, i)

                log_labels_balance(
                    _run,
                    labels=batch[gp.ArrayKey("LABELS")].data,
                    num_classes=_config["data"]["num_classes"],
                    iteration=i,
                )
                # Log metrics training
                log_metrics(
                    _run,
                    target=batch[gp.ArrayKey("LABELS")].data,
                    prediction_probas=batch[gp.ArrayKey("PREDICTIONS")].data,
                    mask=batch[gp.ArrayKey("MASK")].data,
                    metric_mask=batch[gp.ArrayKey("METRIC_MASK")].data,
                    iteration=i,
                    mode="train",
                )

            if i % _config["validation"]["validate_every"] == 0:
                model.eval()

                # loop over validation objects; each one has a pipeline
                for val_idx, validation in enumerate(validations):
                    # current validation pipeline
                    val_p = validation.pipeline

                    val_request = gp.BatchRequest()
                    provider_spec = validation.scan.spec
                    for key, spec in provider_spec.items():
                        if key in validation.request:
                            request_spec = spec.copy()
                            request_spec.dtype = None
                            val_request[key] = request_spec

                    val_batch = val_p.request_batch(val_request)

                    val_losses = np.atleast_1d(val_batch.loss)
                    for l_i, l in enumerate(val_losses):
                        _run_dummy.log_scalar(f"loss_val_ds_{val_idx}_type_{l_i}", l, i)

                    # TODO there is no single validation loss to save any more.
                    # Extend to multiclass.
                    # validation_loss = val_losses[0]

                    if i % _config["sacred"]["log_every"] == 0:
                        log_metrics(
                            _run,
                            target=val_batch[gp.ArrayKey("LABELS")].data,
                            prediction_probas=val_batch[
                                gp.ArrayKey("PREDICTIONS")
                            ].data,
                            mask=val_batch[gp.ArrayKey("MASK")].data,
                            metric_mask=val_batch[gp.ArrayKey("METRIC_MASK")].data,
                            iteration=i,
                            mode=f"validation_ds_{val_idx}",
                        )
                model.train()

        # release (teardown) validation pipelines
        logger.debug("tearing down val pipelines")
        for idx_pipeline, validation in enumerate(validations):
            validations[idx_pipeline].pipeline.internal_teardown()
        logger.debug("tear down completed")

    return log_result(_run_dummy=_run, _config=_config, metric_val=validation_loss)


# def get_config_from_db(url, db_name, run_id):
#     with MongoClient(host=url, port=27017) as client:
#         db = client[db_name]
#         run_document = db['runs'].find_one({'_id': run_id})
#
#     return run_document['config']


def load_run(run_id):
    # Check if the previous run ID is in training_runs or mock_db
    ledger_path = Path(__file__).resolve().parents[3].joinpath("mock_db/ledger.json")
    with open(f"{ledger_path}") as f:
        ledger = json.load(f)
    assert str(run_id) in ledger.keys(), "Run ID not found in ~/incasem/mock_db"

    mock_db_path = Path(__file__).resolve().parents[3].joinpath("mock_db")
    with open(f"{mock_db_path}/{ledger[str(run_id)]}") as f:
        config = json.load(f)

    return config


def parse_argmuents():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--repeat_run", type=int, help="Run ID of training to repeat.")
    parser.add_argument(
        "--continue_run", type=int, help="Run ID of training to continue"
    )
    parser.add_argument(
        "--start_from",
        nargs=2,
        metavar=("RUN ID", "checkpoint"),
        help="Start training from a previous trained model, \
             given its run ID and checkpoint",
    )

    args, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining_argv]

    if args.repeat_run is not None:
        config = load_run(args.repeat_run)

    elif args.continue_run is not None:
        config = load_run(args.continue_run)

        config["training"]["continue_id"] = args.continue_run

    elif args.start_from is not None:
        prev_model_id, _ = args.start_from
        config = load_run(int(prev_model_id))
        config["training"]["start_from"] = args.start_from

    else:
        config = None

    return config, args, remaining_argv


if __name__ == "__main__":
    config, args, remaining_argv = parse_argmuents()

    curr_path = Path(__file__).resolve().parent
    path_to_conf = curr_path.joinpath("config_training.yaml")
    if config is not None:
        logger.debug(config)
    else:
        config = {}
        with open(f"{path_to_conf}", "r") as file:
            yaml_data = yaml.safe_load(file)

    val_arg_dict = {}
    train_arg_dict = {}
    torch_arg_dict = {}

    if "--name" in remaining_argv:
        name_idx = remaining_argv.index("--name") + 1
        name = remaining_argv[name_idx]
        config["name"] = name

    for item in remaining_argv:
        if "training." in item and "=" in item:
            pattern = r"\btraining\.(\S+)\s*=\s*(\S+)\b"
            # Find all matches in the text
            matches = re.findall(pattern, item)
            if len(matches) > 0:
                k, v = item.split("training.")[-1].split("=")
                train_arg_dict[k] = v

        if "validation." in item and "=" in item:
            pattern = r"\bvalidation\.(\S+)\s*=\s*(\S+)\b"
            # Find all matches in the text
            matches = re.findall(pattern, item)
            if len(matches) > 0:
                k, v = item.split("validation.")[-1].split("=")
                val_arg_dict[k] = v

        if "torch." in item and "=" in item:
            pattern = r"\btorch\.(\S+)\s*=\s*(\S+)\b"
            # Find all matches in the text
            matches = re.findall(pattern, item)
            if len(matches) > 0:
                k, v = item.split("torch.")[-1].split("=")
                torch_arg_dict[k] = v

    config = {**config, **yaml_data}
    config["training"] = {**config["training"], **train_arg_dict}
    config["validation"] = {**config["validation"], **val_arg_dict}
    config["torch"] = {**config["torch"], **torch_arg_dict}

    _run_dummy = TrainingRunDummy()

    try:
        name = config["name"]
    except KeyError:
        name = "training"

    name = name + f"_{_run_dummy._id}"

    path_to_mock_db = curr_path.parents[1].joinpath("mock_db")
    # mock_db_path = curr_path.joinpath("mock_db")

    # Update ledger
    ledger_json = path_to_mock_db.joinpath("ledger.json")
    with open(ledger_json) as fp:
        ledger = json.load(fp)

    ledger[str(_run_dummy._id)] = name + ".json"

    with open(ledger_json, mode="w") as f:
        json.dump(ledger, f)

    # Write config
    path_to_config = path_to_mock_db.joinpath(f"{name}.json")
    with open(f"{path_to_config}", mode="w") as f:
        json.dump(config, f)

    seed_dummy = 42

    train(config, _run_dummy, seed_dummy)
