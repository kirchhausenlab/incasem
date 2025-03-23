#!/usr/bin/env python
"""Create a segmentation by thresholding a predicted probability map using Dask.

This script replicates the original behavior:
  - It reads a predicted probability map from a Zarr container,
  - Optionally reads a mask from another Zarr container,
  - For each block (defined by a user\u2011provided chunk shape), thresholds the probabilities
    (and, if available, applies the mask),
  - Converts the resulting binary segmentation to type np.uint32 and scales it (0/255),
  - Writes each processed block into a new output dataset in the same container.

Dependencies:
  - python 3.8+
  - dask[distributed]
  - zarr
  - configargparse
  - numpy
  - scikit-image
  - numcodecs
"""

import os
import numpy as np
import configargparse as argparse
import skimage.morphology  # not used here, but might be needed for consistency
from dask.distributed import Client, as_completed
import zarr
from numcodecs import Blosc
import itertools
from tqdm import tqdm
from time import time as now, sleep


# --- Helper: Partition ROI into blocks ---
def partition_roi(total_shape, block_size):
    """
    Partition a volume with shape total_shape into blocks with the given block_size.
    Returns:
      - A list of slice tuples (one per block).
      - The grid dimensions (tuple of ints).
    """
    dims = len(total_shape)
    grid_dims = [(total_shape[d] + block_size[d] - 1) // block_size[d] for d in range(dims)]
    slices = []
    for idx in itertools.product(*(range(n) for n in grid_dims)):
        block_slice = tuple(
            slice(idx[d] * block_size[d], min((idx[d] + 1) * block_size[d], total_shape[d]))
            for d in range(dims)
        )
        slices.append(block_slice)
    return slices, grid_dims


# --- Worker Function ---
def process_segmentation_block(filename, ds_name, mask_filename, mask_ds_name, read_roi, threshold):
    """
    Process one block for segmentation.

    Re-opens the prediction dataset from filename and ds_name,
    and (if provided) the mask from mask_filename and mask_ds_name.

    The segmentation is defined as:
      - If a mask exists: (probas >= threshold) AND (mask != 0)
      - Otherwise: (probas >= threshold)

    The result is converted to np.uint32 and multiplied by 255.

    Parameters:
      filename      : Path to the Zarr container for predictions.
      ds_name       : Name of the prediction dataset.
      mask_filename : Path to the Zarr container for the mask (or empty string if not provided).
      mask_ds_name  : Name of the mask dataset.
      read_roi      : Tuple of slices defining the region to process.
      threshold     : Threshold value (float).

    Returns:
      A numpy array with dtype np.uint32.
    """
    # Re-open prediction dataset
    pred = zarr.open(filename, mode='r')[ds_name]
    probas_block = np.array(pred[read_roi])

    # If a mask filename is provided, try to open and read the mask block
    if mask_filename:
        try:
            mask = zarr.open(mask_filename, mode='r')[mask_ds_name]
            mask_block = np.array(mask[read_roi])
            seg = ((probas_block >= threshold) & (mask_block != 0))
        except Exception:
            seg = (probas_block >= threshold)
    else:
        seg = (probas_block >= threshold)

    segmentation = seg.astype(np.uint32) * 255
    return segmentation


# --- Main Processing Function ---
def extract_segmentation_with_threshold(filename, ds_name, mask_filename, mask_ds_name,
                                        out_ds_name, chunk_shape, threshold, num_workers):
    """
    Create a segmentation by thresholding the predicted probability map.

    - Opens the prediction dataset (and optionally a mask) from the given Zarr container.
    - Creates an output dataset (with the same shape and chunking, using zlib compression).
    - Partitions the prediction ROI into blocks using the given chunk_shape.
    - Submits a task for each block via client.submit.
    - As each task completes, writes the result into the corresponding region in the output dataset.
    """
    # Open input prediction dataset
    pred_zarr = zarr.open(filename, mode='r')
    pred = pred_zarr[ds_name]
    total_shape = pred.shape

    # Open (or create) the output dataset in the same container.
    out_zarr = zarr.open(filename, mode='a')
    if out_ds_name in out_zarr:
        print(f"Deleting existing dataset '{out_ds_name}' in {filename}")
        del out_zarr[out_ds_name]
    out_array = out_zarr.create_dataset(
        name=out_ds_name,
        shape=total_shape,
        chunks=tuple(chunk_shape),
        dtype=np.uint32,
        compressor=Blosc(cname="zlib", clevel=3)
    )

    # Partition ROI
    blocks, grid_dims = partition_roi(total_shape, tuple(chunk_shape))
    total_blocks = len(blocks)
    print(f"Processing volume of shape {total_shape} in {total_blocks} blocks")

    # Create a Dask client.
    client = Client(n_workers=num_workers)
    print(f"Dask client created with {num_workers} workers")
    sleep(2)

    # Schedule each block for processing.
    scheduled = []
    for block in blocks:
        future = client.submit(process_segmentation_block,
                               filename, ds_name,
                               mask_filename, mask_ds_name,
                               block, threshold)
        scheduled.append((block, future))

    # Iterate over futures as they complete.
    progress_bar = tqdm(total=len(scheduled), desc="Processing Blocks")
    mapping = {f: block for (block, f) in scheduled}
    for future in as_completed([f for _, f in scheduled]):
        block = mapping[future]
        result = future.result()
        out_array[block] = result
        progress_bar.update(1)
    progress_bar.close()

    client.close()
    print("Done.")


# --- Command-line Interface ---
def parse_args():
    p = argparse.ArgParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add('--config', is_config_file=True, help='config file path')
    p.add('--prediction_filename', required=True,
          help="Zarr file with the prediction.")
    p.add('--dataset', '-d', required=True,
          help='Name of the dataset with prediction probabilities.')
    p.add('--mask_filename', default="",
          help="Zarr file with the mask.")
    p.add('--mask', '-m', default='volumes/mask',
          help='Binary mask to exclude non-cell voxels.')
    p.add('--out_dataset', '-o', required=True,
          help='Name of the output segmentation in the prediction zarr file.')
    p.add('--chunk_shape', '-c', nargs='+', type=int, default=[128, 128, 128],
          help='Size of a chunk in voxels. Should be a multiple of the existing chunk size.')
    p.add('--threshold', '-t', type=float, required=True,
          help='Threshold for positive prediction.')
    p.add('--num_workers', '-n', type=int, default=32,
          help='Number of dask workers.')
    args = p.parse_args()
    print("\nCommand Line Args:", p.format_values())
    return args


def main():
    args = parse_args()
    extract_segmentation_with_threshold(
        filename=args.prediction_filename,
        ds_name=args.dataset,
        mask_filename=args.mask_filename,
        mask_ds_name=args.mask,
        out_ds_name=args.out_dataset,
        chunk_shape=args.chunk_shape,
        threshold=args.threshold,
        num_workers=args.num_workers
    )


if __name__ == '__main__':
    main()
