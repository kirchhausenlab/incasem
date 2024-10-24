import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from shutil import copyfile

import gunpowder as gp
import numpy as np
import torch
import yaml
from loguru import logger
import wandb
import hydra
from typing import Optional, Union, Tuple, List, Dict, Any
from omegaconf import DictConfig
import incasem as fos

import torch.distributed as dist


def setup_wandb(
    cfg: DictConfig,
    hyperparameters: Dict[str, Any],
) -> wandb.run:
    run = wandb.init(
        project=cfg.logging.wandb_config.project,
        entity=cfg.logging.wandb_config.entity,
        config=hyperparameters,
        name=cfg.logging.wandb_config.run_name,
        resume=cfg.logging.wandb_config.resume,
    )
    return run


def _log(
    run: wandb.run,
    metrics: Dict[str, Any],
    **kwargs: Any,
) -> None:
    logger.info("Logging metrics: %s" % metrics)
    run.log(metrics, **kwargs)


def setup(
    rank: int,
    world_size: int,
) -> None:
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def torch_backend_setup() -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = False


def cleanup() -> None:
    dist.destroy_process_group()


def sync() -> None:
    dist.barrier()


def setup_torch(
    cfg: DictConfig,
) -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # if rank and world_size are not set, set default values
    if rank is None:
        rank = 0
    if world_size is None:
        world_size = 1

    setup(rank=rank, world_size=world_size)
    torch_backend_setup()
    if cfg.device == "cpu":
        logger.info("Using CPU, line 55, setup_torch")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        logger.info(f"Using GPU {cfg.device}, line 58, setup_torch")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device)


def model_setup(
    cfg: DictConfig,
) -> torch.nn.Module:
    try:
        model_type = cfg.model_type
        if model_type == "UNet":
            model = fos.torch.models.Unet(
                in_channels=cfg.model.unet.in_channels,
                num_fmaps=cfg.model.unet.num_fmaps,
                fmap_inc_factor=cfg.model.unet.fmap_inc_factor,
                downsample_factors=tuple(
                    tuple(i) for i in cfg.model.unet.downsample_factors
                ),
                activation=str(cfg.model.unet.activation),
                voxel_size=cfg.data.voxel_size,
                num_fmaps_out=cfg.model.unet.num_fmaps_out,
                num_heads=cfg.model.unet.num_heads,
                constant_upsample=cfg.model.unet.constant_upsample,
                padding=cfg.model.unet.padding,
            )
        device = cfg.device
        model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(
            f"Total parameters: {total_params} and Trainable parameters: {trainable_params}"
        )
        return model
    except Exception as e:
        logger.error("Error in model_setup: %s" % e)
        raise e


def loss_setup(
    cfg: DictConfig,
) -> None:
    try:
        weight = torch.tensor(list(cfg.model.unet.loss.weight), dtype=torch.float32)
        loss_type = cfg.loss_type
        if loss_type == "CrossEntropyLoss":
            loss = fos.torch.loss.CrossEntropyLossWithScalingAndMeanReduction(
                weight=weight, device=cfg.device
            )
        return loss

    except Exception as e:
        logger.error("Error in loss_setup: %s" % e)
        raise e


def training_setup(
    cfg: DictConfig,
    model: torch.nn.Module,
):
    try:
        loss = loss_setup(cfg)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(cfg.train.optimizer.lr),
            weight_decay=float(cfg.train.optimizer.weight_decay),
        )

        pipeline_type = {
            "baseline_with_context": fos.pipeline.TrainingBaselineWithContext
        }[cfg.train.pipeline]

        training = pipeline_type(
            data_config=cfg.train.data,
            run_dir=cfg.directories.runs,
            run_path_prefix=os.path.expanduser(cfg.directories.runs),
            data_path_prefix=os.path.expanduser(cfg.directories.data),
            model=model,
            loss=loss,
            optimizer=optimizer,
            num_classes=int(cfg.data.num_classes),
            voxel_size=cfg.data.voxel_size,
            input_size_voxels=cfg.train.input_size_voxels,
            output_size_voxels=cfg.train.output_size_voxels,
            reject_probability=float(cfg.train.reject.reject_probability),
            reject_min_masked=float(cfg.train.reject.reject_min_masked),
            random_seed=int(cfg.seed),
        )

        device = cfg.device
        training.train_node.gpus = [] if device == "cpu" else [int(device)]
        training.train_node.save_every = int(cfg.train.save_every)
        training.train_node.log_every = int(cfg.train.log_every)
        training.downsample.factor = int(cfg.data.downsample_factor)

        try:
            training.balance_labels.clipmin = float(
                cfg.model.unet.loss.balance_labels.clipmin
            )
            training.balance_labels.clipmax = float(
                cfg.model.unet.loss.balance_labels.clipmax
            )
        except AttributeError as e:
            logger.warning(
                "Trying to set BalanceLabels attributes, but it is not used. %s" % e
            )
            cfg.model.unet.loss.balance_labels.clipmin = None
            cfg.model.unet.loss.balance_labels.clipmax = None

        try:
            training.augmentation.nodes[
                "simple_0"
            ].transpose_only = cfg.train.augmentation.simple_0.transpose_only
        except (KeyError, AttributeError) as e:
            logger.warning(
                "Trying to set Augmentation attributes, but it is not used. %s" % e
            )
            cfg.train.augmentation.simple_0.transpose_only = None

        try:
            training.augmentation.nodes["elastic"].control_point_spacing = tuple(
                cfg.train.augmentation.elastic.control_point_spacing
            )
            training.augmentation.nodes["elastic"].jitter_sigma = tuple(
                cfg.train.augmentation.elastic.jitter_sigma
            )
            training.augmentation.nodes["elastic"].subsample = int(
                cfg.train.augmentation.elastic.subsample
            )
        except (KeyError, AttributeError):
            logger.warning(
                "Trying to set parameters for ElasticAugment, but it is not used."
            )

        try:
            training.augmentation.nodes[
                "simple_1"
            ].transpose_only = cfg.train.augmentation.simple.transpose_only
        except (KeyError, AttributeError):
            logger.warning("SimpleAugment 1 transpose only not set.")

        try:
            training.augmentation.nodes["intensity"].scale_min = 1.0 - float(
                cfg.train.augmentation.intensity.scale
            )
            training.augmentation.nodes["intensity"].scale_max = 1.0 + float(
                cfg.train.augmentation.intensity.scale
            )
            training.augmentation.nodes["intensity"].shift_min = -1.0 * float(
                cfg.train.augmentation.intensity.shift
            )
            training.augmentation.nodes["intensity"].shift_max = float(
                cfg.train.augmentation.intensity.shift
            )
        except (KeyError, AttributeError):
            logger.warning(
                "Trying to set parameters for IntensityAugment, but it is not used."
            )

        try:
            training.precache.cache_size = int(cfg.train.precache.cache_size)
            training.precache.num_workers = int(cfg.train.precache.num_workers)
        except AttributeError:
            logger.warning("Trying to set Precache attributes, but it is not used.")
        except KeyError:
            logger.warning(
                "Trying to set Precache attributes, but not specified in config."
            )

        training.snapshot.every = int(cfg.train.snapshot.every)
        training.profiling_stats.every = int(cfg.train.profiling_stats.every)

        return training
    except Exception as e:
        logger.error("Error in training_setup: %s" % e)
        raise e


def multiple_validation_setup(
    cfg: DictConfig,
    model: torch.nn.Module,
) -> List[fos.pipeline.ValidationBaselineWithContext]:
    val_datasets = fos.utils.create_multiple_config(cfg.validate.data)
    validations = []
    for val_ds in val_datasets:
        validations.append(validation_setup(cfg=cfg, model=model, val_dataset=val_ds))
    return validations


def validation_setup(
    val_dataset,
    cfg: DictConfig,
    model: torch.nn.Module,
):
    loss = loss_setup(cfg)

    pipeline_type = {
        "baseline_with_context": fos.pipeline.ValidationBaselineWithContext
    }[cfg.validate.pipeline]

    validation = pipeline_type(
        data_config=val_dataset,
        run_dir=cfg.directories.runs,
        run_path_prefix=os.path.expanduser(cfg.directories.runs),
        data_path_prefix=os.path.expanduser(cfg.directories.data),
        model=model,
        loss=loss,
        num_classes=int(cfg.data.num_classes),
        voxel_size=cfg.data.voxel_size,
        input_size_voxels=cfg.validate.input_size_voxels,
        output_size_voxels=cfg.validate.output_size_voxels,
        run_every=int(cfg.validate.run_every),
        random_seed=cfg.seed,
    )
    device = cfg.device
    validation.predict.gpus = [] if device == "cpu" else [int(device)]
    validation.downsample.factor = int(cfg.data.downsample_factor)
    # Balance Labels
    try:
        validation.balance_labels.clipmin = float(
            cfg.model.unet.loss.balance_labels.clipmin
        )
        validation.balance_labels.clipmax = float(
            cfg.model.unet.loss.balance_labels.clipmax
        )
    except AttributeError as e:
        logger.warning(
            "Trying to set BalanceLabels attributes, but it is not used. %s" % e
        )
        cfg.model.unet.loss.balance_labels.clipmin = None
        cfg.model.unet.loss.balance_labels.clipmax = None

    validation.snapshot.every = int(cfg.validate.snapshot.every)

    return validation


def log_metrics(
    run: wandb.run,
    target: np.ndarray,
    prediction_probas: np.ndarray,
    mask: np.ndarray,
    metric_mask: np.ndarray,
    iteration: int,
    mode: str,
) -> None:
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
        run.log_scalar(f"dice_class_{label}_{mode}", score, iteration)
        logger.info(f"{mode} | Dice score class {label}: {score}")


def _log_metrics(_run, target, prediction_probas, mask, metric_mask, iteration, mode):
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
    setup_torch(cfg=_config)
    model = model_setup(cfg=_config)
    training = training_setup(cfg=_config, model=model)
    validations = multiple_validation_setup(
        cfg=_config,
        model=model,
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


config_path = Path(__file__).resolve().parents[2].joinpath("configs")


@hydra.main(version_base=None, config_path=str(config_path), config_name="config.yaml")
def main(cfg: DictConfig):
    print(cfg)

    setup_torch(cfg)
    model = model_setup(cfg)
    training = training_setup(cfg, model)
    logger.info(f"Training pipeline: {training.pipeline}")
    runs = setup_wandb(cfg, hyperparameters={})
    validations = multiple_validation_setup(cfg, model)
    logger.info(f"Validation pipelines: {validations}")
    sync()
    cleanup()


if __name__ == "__main__":
    main()
