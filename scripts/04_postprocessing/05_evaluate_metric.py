"""
Evaluate segmentation metric for a single probability map with different
thresholds.
"""

import logging
from time import time as now

import numpy as np
import configargparse as argparse

from funlib.persistence import Array, open_ds, prepare_ds
from funlib.geometry import Roi, Coordinate

import incasem as fos


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logging.getLogger(
    'incasem.metrics.precision_recall'
).setLevel(logging.WARNING)


def split_zarr_path(path):
    if path is None:
        return None
    filename, extension, ds_name = path.rpartition('.zarr/')
    filename = (filename + extension).rstrip('/')
    ds_name = ds_name.rstrip('/')
    return filename, ds_name


def evaluate_metric(
        metric,
        labels_path,
        prediction_probas_path,
        mask_path,
        metric_mask_path,
        roi_padding,
        thresholds,
        # num_workers
):
    labels = open_ds(
        *split_zarr_path(labels_path),
        mode='r'
    )

    probas = open_ds(
        *split_zarr_path(prediction_probas_path),
        mode='r'
    )

    if not labels.roi.contains(probas.roi):
        raise ValueError((
            f"The labels {labels.roi} do not suffice to evaluate "
            "predictions in {probas.roi}."
        ))

    try:
        mask = open_ds(
            *split_zarr_path(mask_path),
            mode='r'
        )
        if not mask.roi.contains(probas.roi):
            raise ValueError((
                f"The provided mask {mask.roi} does not cover the predictions "
                " {probas.roi}."
            ))
    except TypeError:
        logger.warning((
            "Did not find a mask dataset "
            f"at {mask_path}."
        ))
        mask = None

    try:
        metric_mask = open_ds(
            *split_zarr_path(metric_mask_path),
            mode='r'
        )
        if not metric_mask.roi.contains(probas.roi):
            raise ValueError((
                f"The provided metric mask {metric_mask.roi} does not cover"
                " the predictions {probas.roi}."
            ))
    except TypeError:
        logger.warning((
            "Did not find a metric mask dataset "
            f"at {metric_mask_path}."
        ))
        metric_mask = None

    # Remove the zero padding from the predictions
    logger.debug(f"Full Roi: {probas.roi}")
    if len(roi_padding) == 1:
        roi_padding = roi_padding * probas.roi.dims()
    roi_padding = Coordinate(roi_padding) * probas.voxel_size
    roi = probas.roi.grow(-roi_padding, -roi_padding)
    logger.debug(f"Roi without padding: {roi}\n")

    start = now()

    logger.info("Loading data ...")

    labels = labels[roi].to_ndarray()
    # binarize labels
    labels = (labels != 0).astype(np.uint8)

    probas = probas[roi].to_ndarray()

    if metric_mask is not None:
        metric_mask = metric_mask[roi].to_ndarray()
    else:
        metric_mask = np.ones_like(labels, dtype=np.uint8)

    if mask is not None:
        logger.info("Masking probabilities ...")
        mask = mask[roi].to_ndarray()
        probas = probas * (mask != 0).astype(probas.dtype)

        metric_mask = np.logical_and(
            mask.astype(bool),
            metric_mask.astype(bool)
        )

    # TODO parallelize, log results to DB

    scores_and_thresholds = []
    for thres in thresholds:
        start_threshold = now()
        if metric in ['dice', 'jaccard']:
            score = \
                fos.metrics.pairwise_distance_metric_thresholded(
                    target=labels,
                    prediction_probas=np.array(
                        [np.zeros_like(probas), probas]),
                    metric=metric,
                    threshold=thres,
                    foreground_class=1,
                    mask=metric_mask,
                )
            score_aggregated = score
        elif metric == 'precision_recall':
            _, score = fos.metrics.precision_recall(
                target=labels,
                prediction_probas=np.array(
                    [1 - probas, probas]),
                mask=metric_mask,
                threshold=thres,
            )
            score_aggregated = score[0] + score[1]
        else:
            raise NotImplementedError(f"Metric {metric} not implemented.")

        scores_and_thresholds.append((score_aggregated, score, thres))
        logger.info((
            f"{metric} at threshold {thres}: "
            f"{score:{'.3f' if isinstance(score, float) else ''}} "
            f"(in {now() - start_threshold:.1f} s)"
        ))

    _, max_score, max_thres = max(scores_and_thresholds)
    logger.info(f"\n\nMax {metric} at threshold {max_thres}: {max_score}")
    logger.info(f"Done in {now() - start} s")


def parse_args():
    p = argparse.ArgParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add('--config', is_config_file=True, help='config file path')
    p.add(
        '--metric',
        default='jaccard',
        # TODO add more options
        choices=['jaccard', 'dice', 'precision_recall'],
        help=(
            'A distance function between two boolean vectors as defined in '
            'scipy.special.distance.'
        )
    )
    p.add(
        '--labels',
        '-l',
        required=True,
    )
    p.add(
        '--prediction_probas',
        '-p',
        required=True,
        help='Name of the dataset with prediction probabilities.'
    )
    p.add(
        '--mask',
        help='Binary mask to predict background for all non-cell voxels.'
    )
    p.add(
        '--metric_mask',
        help='Binary mask to ignore predictions at the boundary of objects'
    )
    p.add(
        '--roi_padding',
        type=int,
        nargs='+',
        default=[46, 46, 46],
        help=(
            'The prediction ROI is not filled at the boundaries. '
            'This empty padding should not affect the metric calculation. '
            'Can be either a single integer or one integer per dimension, '
            'in voxels, zyx.'
        )
    )
    p.add(
        '--threshold_start',
        type=float,
        default=0.5,
        help='Lowest threshold for extracting predictions.'
    )
    p.add(
        '--threshold_stop',
        type=float,
        default=0.5,
        help='Highest threshold for extracting predictions.'
    )
    p.add(
        '--threshold_step',
        type=float,
        default=0.1,
        help='Interval between thresholds for extracting predictions.'
    )
    # p.add(
    # '--num_workers',
    # '-n',
    # type=int,
    # default=32,
    # help='Number of daisy processes.'
    # )

    args = p.parse_args()

    epsilon = 1e-5
    if args.threshold_step <= epsilon:
        raise ValueError(f"Threshold set must be bigger than {epsilon}.")

    args.thresholds = np.arange(
        args.threshold_start,
        args.threshold_stop + epsilon,
        args.threshold_step)

    # logger.info(f'\n{p.format_values()}')
    logger.info('\n')
    logger.info(f"Evaluate {args.metric} for thresholds {args.thresholds}")

    return args


def main():
    args = parse_args()
    evaluate_metric(
        metric=args.metric,
        labels_path=args.labels,
        prediction_probas_path=args.prediction_probas,
        mask_path=args.mask,
        metric_mask_path=args.metric_mask,
        roi_padding=args.roi_padding,
        thresholds=args.thresholds,
        # num_workers=args.num_workers
    )


if __name__ == '__main__':
    main()