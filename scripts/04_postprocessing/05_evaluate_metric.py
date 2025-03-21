#!/usr/bin/env python
"""
Evaluate segmentation metric for a single probability map with different thresholds.

This script:
  - Loads labels, prediction probabilities, and (optionally) mask and metric mask
    from Zarr containers.
  - Computes a cropped ROI by removing a user-specified padding (in voxels) from the full ROI.
  - Loads the data (and applies masking if provided).
  - For a set of thresholds, evaluates a segmentation metric (dice/jaccard or precision_recall)
    using incasem.fos.metrics.
  - Parallelizes the threshold loop using Dask.

Dependencies:
  - python 3.8+
  - numpy
  - configargparse
  - zarr
  - dask[distributed]
  - incasem (fos)

No funlib or daisy is used.
"""

import logging
from time import time as now
import numpy as np
import configargparse as argparse
import zarr
from dask.distributed import Client, as_completed
import dask
import incasem as fos
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logging.getLogger('incasem.metrics.precision_recall').setLevel(logging.WARNING)


def split_zarr_path(path):
    if path is None:
        return None
    filename, extension, ds_name = path.rpartition('.zarr/')
    filename = (filename + extension).rstrip('/')
    ds_name = ds_name.rstrip('/')
    return filename, ds_name


# --- ROI Helpers ---
def get_roi(ds):
    """Return ROI as (offset, shape) from ds.attrs if available, else full volume."""
    try:
        roi = ds.attrs["roi"]  # expect a two-element list/tuple: [offset, shape]
        return tuple(roi[0]), tuple(roi[1])
    except KeyError:
        return (0,) * len(ds.shape), ds.shape


def get_voxel_size(ds):
    """Return voxel size from ds.attrs or default to ones."""
    return ds.attrs.get("voxel_size", (1,) * len(ds.shape))


def multiply_tuple(a, b):
    """Element-wise multiplication of two tuples."""
    return tuple(a[i] * b[i] for i in range(len(a)))


def grow_roi(roi, pad):
    """
    Grow (or shrink) ROI by pad. Here, pad is subtracted from the shape.
    For cropping, we want to shrink the ROI:
      new_offset = offset + pad
      new_shape = shape - 2*pad
    """
    offset, shape = roi
    new_offset = tuple(offset[i] + pad[i] for i in range(len(offset)))
    new_shape = tuple(shape[i] - 2 * pad[i] for i in range(len(shape)))
    return (new_offset, new_shape)


def roi_to_slices(roi):
    """Convert an ROI (offset, shape) to a tuple of slices."""
    offset, shape = roi
    return tuple(slice(offset[i], offset[i] + shape[i]) for i in range(len(offset)))


# --- Parallel Evaluation ---
def evaluate_threshold(thres, metric, labels, probas, metric_mask):
    """
    Evaluate a single threshold using fos.metrics.
    """
    if metric in ['dice', 'jaccard']:
        score = fos.metrics.pairwise_distance_metric_thresholded(
            target=labels,
            prediction_probas=np.array([np.zeros_like(probas), probas]),
            metric=metric,
            threshold=thres,
            foreground_class=1,
            mask=metric_mask,
        )
        score_aggregated = score
    elif metric == 'precision_recall':
        _, score = fos.metrics.precision_recall(
            target=labels,
            prediction_probas=np.array([1 - probas, probas]),
            mask=metric_mask,
            threshold=thres,
        )
        score_aggregated = score[0] + score[1]
    else:
        raise NotImplementedError(f"Metric {metric} not implemented.")
    return score_aggregated, score, thres


def evaluate_metric(metric, labels_path, prediction_probas_path, mask_path, metric_mask_path, roi_padding, thresholds):
    # Open labels and prediction datasets using zarr.
    labels_file, labels_ds_name = split_zarr_path(labels_path)
    probas_file, probas_ds_name = split_zarr_path(prediction_probas_path)
    labels_ds = zarr.open(labels_file, mode='r')[labels_ds_name]
    probas_ds = zarr.open(probas_file, mode='r')[probas_ds_name]

    # Ensure labels ROI covers prediction ROI.
    labels_roi = get_roi(labels_ds)
    probas_roi = get_roi(probas_ds)
    if labels_roi != probas_roi:
        raise ValueError(f"The labels ROI {labels_roi} does not match the prediction ROI {probas_roi}.")

    # Try to open mask.
    mask = None
    try:
        mask_file, mask_ds_name = split_zarr_path(mask_path)
        mask = zarr.open(mask_file, mode='r')[mask_ds_name]
        if get_roi(mask) != probas_roi:
            raise ValueError(f"Mask ROI {get_roi(mask)} does not cover predictions {probas_roi}.")
    except (TypeError, KeyError, RuntimeError):
        logger.warning(f"Did not find a mask dataset at {mask_path}.")
        mask = None

    # Try to open metric_mask.
    metric_mask = None
    try:
        mm_file, mm_ds_name = split_zarr_path(metric_mask_path)
        metric_mask = zarr.open(mm_file, mode='r')[mm_ds_name]
        if get_roi(metric_mask) != probas_roi:
            raise ValueError(f"Metric mask ROI {get_roi(metric_mask)} does not cover predictions {probas_roi}.")
    except (TypeError, KeyError, RuntimeError):
        logger.warning(f"Did not find a metric mask dataset at {metric_mask_path}.")
        metric_mask = None

    # Compute ROI padding.
    dims = len(probas_ds.shape)
    if len(roi_padding) == 1:
        roi_padding = tuple(roi_padding[0] for _ in range(dims))
    voxel_size = get_voxel_size(probas_ds)
    padding_pixels = multiply_tuple(roi_padding, voxel_size)
    full_roi = get_roi(probas_ds)
    cropped_roi = grow_roi(full_roi, padding_pixels)
    slices = roi_to_slices(cropped_roi)
    logger.debug(f"Full ROI: {full_roi}")
    logger.debug(f"ROI without padding: {cropped_roi}")

    start = now()
    logger.info("Loading data ...")
    labels_arr = labels_ds[slices]
    labels_arr = (labels_arr != 0).astype(np.uint8)
    probas_arr = probas_ds[slices]
    if metric_mask is not None:
        metric_mask_arr = metric_mask[slices]
    else:
        metric_mask_arr = np.ones_like(labels_arr, dtype=np.uint8)
    if mask is not None:
        logger.info("Masking probabilities ...")
        mask_arr = mask[slices]
        probas_arr = probas_arr * (mask_arr != 0).astype(probas_arr.dtype)
        metric_mask_arr = np.logical_and(mask_arr.astype(bool), metric_mask_arr.astype(bool))
    logger.info("Data loaded.")

    # Evaluate thresholds in parallel.
    client = Client()
    tasks = []
    for th in thresholds:
        tasks.append(dask.delayed(evaluate_threshold)(th, metric, labels_arr, probas_arr, metric_mask_arr))
    futures = client.compute(tasks)
    results = []
    progress_bar = tqdm(total=len(futures), desc="Evaluating thresholds")
    for future in as_completed(futures):
        results.append(future.result())
        progress_bar.update(1)
    progress_bar.close()
    client.close()

    # Select the threshold with the maximum aggregated score.
    _, max_score, max_thres = max(results, key=lambda x: x[0])
    logger.info(f"\n\nMax {metric} at threshold {max_thres}: {max_score}")
    logger.info(f"Done in {now() - start:.2f} s")


def parse_args():
    p = argparse.ArgParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add('--config', is_config_file=True, help='config file path')
    p.add('--metric', default='jaccard', choices=['jaccard', 'dice', 'precision_recall'],
          help='Metric to evaluate.')
    p.add('--labels', '-l', required=True)
    p.add('--prediction_probas', '-p', required=True,
          help='Name of the dataset with prediction probabilities.')
    p.add('--mask', help='Binary mask to predict background for all non-cell voxels.')
    p.add('--metric_mask', help='Binary mask to ignore predictions at the boundary of objects')
    p.add('--roi_padding', type=int, nargs='+', default=[46, 46, 46],
          help='Empty padding around the prediction ROI (in voxels, zyx).')
    p.add('--threshold_start', type=float, default=0.5,
          help='Lowest threshold for extracting predictions.')
    p.add('--threshold_stop', type=float, default=0.5,
          help='Highest threshold for extracting predictions.')
    p.add('--threshold_step', type=float, default=0.1,
          help='Interval between thresholds for extracting predictions.')
    args = p.parse_args()
    epsilon = 1e-5
    if args.threshold_step <= epsilon:
        raise ValueError(f"Threshold step must be bigger than {epsilon}.")
    args.thresholds = np.arange(args.threshold_start, args.threshold_stop + epsilon, args.threshold_step)
    logger.info(f"\nEvaluate {args.metric} for thresholds {args.thresholds}")
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
    )


if __name__ == '__main__':
    main()
