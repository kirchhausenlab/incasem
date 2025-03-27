#!/usr/bin/env python
"""Create a foreground mask for raw EM Data using dask.

This script replicates the original logic that used daisy and funlib by:
  - Reading a raw dataset from a Zarr container,
  - Partitioning it into blocks (using a user\u2011provided chunk shape),
  - For each block: thresholding the data, applying binary opening and closing,
    removing small holes and objects, converting to uint8, and scaling to 0-255,
  - Writing each processed block into a new output dataset in the same container.

Dependencies:
  - python 3.8+
  - dask[distributed]
  - zarr
  - configargparse
  - scikit-image
  - numcodecs
  - numpy
"""

import os
import numpy as np
import configargparse as argparse
import skimage.morphology
from skimage.morphology import (
    ball,
    binary_opening,
    binary_closing,
    remove_small_holes,
    remove_small_objects,
)
from dask.distributed import Client, as_completed
import zarr
from numcodecs import Blosc
import itertools
from tqdm import tqdm
from time import time as now, sleep

# Monkey-patch os.makedirs due to a bug in zarr
_prev_makedirs = os.makedirs


def makedirs(name, mode=0o777, exist_ok=False):
    return _prev_makedirs(name, mode, exist_ok=True)


os.makedirs = makedirs


def process_mask_block(filename, ds_name, read_roi, min_gray_value, max_gray_value):
    """
    Worker function that processes a block from the raw dataset.

    Parameters:
      - filename: Path to the Zarr container.
      - ds_name: Name of the raw dataset.
      - read_roi: A tuple of slices defining the region to read.
      - min_gray_value, max_gray_value: Threshold values.

    Returns:
      - A numpy array (dtype np.uint8) with values scaled to 0-255.
    """
    # Re-open the raw dataset in read-only mode.
    raw = zarr.open(filename, mode="r")[ds_name]
    data = np.array(raw[read_roi])

    # Apply threshold: only pixels between min and max remain.
    mask = (data > min_gray_value) & (data < max_gray_value)
    # Use 'footprint' instead of 'selem'
    mask = binary_opening(mask, footprint=ball(3))
    mask = binary_closing(mask, footprint=ball(2))
    mask = remove_small_holes(mask, area_threshold=100000)
    mask = remove_small_objects(mask, min_size=100000)
    mask = mask.astype(np.uint8)
    mask *= 255
    return mask


def partition_roi(total_shape, block_size):
    """
    Partition a total shape (tuple) into a list of block slices given block_size.

    Returns:
      - A list of slice tuples (one per block).
      - The grid dimensions (tuple of ints).
    """
    dims = len(total_shape)
    grid_dims = [
        (total_shape[d] + block_size[d] - 1) // block_size[d] for d in range(dims)
    ]
    slices = []
    for idx in itertools.product(*(range(n) for n in grid_dims)):
        block_slice = tuple(
            slice(
                idx[d] * block_size[d],
                min((idx[d] + 1) * block_size[d], total_shape[d]),
            )
            for d in range(dims)
        )
        slices.append(block_slice)
    return slices, grid_dims


def create_mask(
    filename,
    ds_name,
    out_ds_name,
    chunk_shape,
    min_gray_value,
    max_gray_value,
    num_workers,
):
    """
    Create the foreground mask using dask:

    - Opens the raw dataset from the given Zarr container.
    - Creates a new output dataset (with the same shape and chunking, and using zlib compression).
    - Partitions the raw dataset into blocks (using the provided chunk_shape).
    - Submits a task for each block via client.submit.
    - As each block completes, writes its result into the corresponding region of the output dataset.
    """
    # Open input Zarr dataset.
    raw_zarr = zarr.open(filename, mode="r")
    raw = raw_zarr[ds_name]
    total_shape = raw.shape

    # Create (or overwrite) the output Zarr dataset in the same container.
    out_zarr = zarr.open(filename, mode="a")
    if out_ds_name in out_zarr:
        print(f"Deleting existing dataset '{out_ds_name}' in {filename}")
        del out_zarr[out_ds_name]
    out_array = out_zarr.create_dataset(
        name=out_ds_name,
        shape=total_shape,
        chunks=tuple(chunk_shape),
        dtype=np.uint8,
        compressor=Blosc(cname="zlib", clevel=3),
    )

    # Partition the volume into blocks.
    blocks, grid_dims = partition_roi(total_shape, tuple(chunk_shape))
    total_blocks = len(blocks)
    print(f"Processing volume of shape {total_shape} in {total_blocks} blocks")

    # Create a Dask client.
    client = Client(n_workers=num_workers)
    print(f"Dask client created with {num_workers} workers")
    sleep(2)

    # Schedule each block for processing using client.submit.
    scheduled = []
    for block in blocks:
        # Here, read_roi == write_roi == block.
        scheduled.append((
            block,
            client.submit(
                process_mask_block,
                filename,
                ds_name,
                block,
                min_gray_value,
                max_gray_value,
            ),
        ))

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


def parse_args():
    p = argparse.ArgParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add("--config", is_config_file=True, help="config file path")
    p.add("--filename", "-f", required=True, help="Input zarr container filename")
    p.add(
        "--dataset",
        "-d",
        required=True,
        help="Name of the raw dataset in the zarr container",
    )
    p.add(
        "--out_dataset",
        "-o",
        default="volumes/mask",
        help="Name of the new mask dataset",
    )
    p.add(
        "--chunk_shape",
        "-c",
        nargs="+",
        type=int,
        default=[128, 128, 128],
        help="Size of a chunk in voxels. Should be a multiple of the existing chunk size.",
    )
    p.add(
        "--min_gray_value",
        type=int,
        default=2,
        help="Lower boundary for masking by value",
    )
    p.add(
        "--max_gray_value",
        type=int,
        default=180,
        help="Upper boundary for masking by value",
    )
    p.add("--num_workers", "-n", type=int, default=32, help="Number of dask workers")
    args = p.parse_args()
    print("\nCommand Line Args:", p.format_values())
    return args


def main():
    args = parse_args()
    create_mask(
        args.filename,
        args.dataset,
        args.out_dataset,
        args.chunk_shape,
        args.min_gray_value,
        args.max_gray_value,
        args.num_workers,
    )


if __name__ == "__main__":
    main()
