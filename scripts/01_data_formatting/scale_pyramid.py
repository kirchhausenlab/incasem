#!/usr/bin/env python
"""
Scale Pyramid for Neuroglancer using Dask

This script builds a multi-resolution pyramid from a raw Zarr dataset.
It creates new downscaled datasets (s0, s1, s2, …) in the same container so that the data
can be efficiently viewed with neuroglancer.

Behavior is identical to the original:
  - If the input dataset name does not end with '/s0', it is renamed.
  - For each scale, the new voxel size, ROI, and chunk sizes are computed,
    a new dataset is created, and the data is downscaled block‐wise in parallel.
  - The “shrink” behavior is implemented so that boundary blocks are shrunk.

Dependencies:
  - python 3.8+
  - numpy
  - argparse
  - dask[distributed]
  - zarr
  - numcodecs
  - skimage.measure
  - tqdm
"""

import os
import numpy as np
import argparse
import dask
from dask.distributed import Client, as_completed
import zarr
from numcodecs import Blosc
import skimage.measure
from time import time as now, sleep
import itertools
from tqdm import tqdm

# Monkey-patch os.makedirs due to a bug in zarr
_prev_makedirs = os.makedirs


def makedirs(name, mode=0o777, exist_ok=False):
    return _prev_makedirs(name, mode, exist_ok=True)


os.makedirs = makedirs


# --- Minimal coordinate helpers (mimicking funlib behavior) ---

def snap_to_grid(roi, voxel_size):
    """
    Given a ROI (offset, shape) and voxel_size (tuple),
    grow the shape so that each dimension is a multiple of voxel_size.
    """
    offset, shape = roi
    new_shape = tuple(((shape[d] + voxel_size[d] - 1) // voxel_size[d]) * voxel_size[d]
                      for d in range(len(shape)))
    return (offset, new_shape)


def multiply_tuple(a, b):
    """Element-wise multiplication of two tuples."""
    return tuple(a[d] * b[d] for d in range(len(a)))


# --- Downscaling Functions ---

def partition_roi(total_shape, block_size):
    """
    Partition a total shape (tuple) into a list of block slices given block_size.
    Returns a list of slice tuples and the grid dimensions.
    (This implements the 'shrink' fit: the last block is shrunk to fit.)
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


# NEW: Revised downscale_block which uses the provided block_read as computed in downscale_dask.
def downscale_block(in_array, out_array, factor, block_read, block_write):
    """
    Downscale one block.

    block_read: the input region computed as:
         slice(block_write.start * factor, block_write.start * factor + (block_length)*factor)
    block_write: the output region.

    This function checks that the read region has exactly the desired size (if not, pads it),
    then applies downscaling (via slicing for labels or block_reduce with np.mean) so that the
    resulting shape exactly equals the shape of block_write.
    """
    dims = len(block_write)
    # Desired size in each dimension: (write_length * factor)
    desired = [(block_write[d].stop - block_write[d].start) * factor[d] for d in range(dims)]
    # Use the provided block_read directly.
    read_region = in_array[block_read]
    read_region = np.array(read_region)
    # If read_region is smaller than desired, pad on the right.
    pad_width = []
    for d in range(dims):
        current = read_region.shape[d]
        pad_right = max(0, desired[d] - current)
        pad_width.append((0, pad_right))
    if any(p != (0, 0) for p in pad_width):
        read_region = np.pad(read_region, pad_width, mode='constant', constant_values=0)

    # Downscale.
    if in_array.dtype in [np.uint64, np.uint32]:
        # For labels: use slicing with offset.
        slices_down = tuple(slice(f // 2, None, f) for f in factor)
        out_data = read_region[slices_down]
    else:
        out_data = skimage.measure.block_reduce(read_region, factor, np.mean)

    expected_shape = tuple(s.stop - s.start for s in block_write)
    if out_data.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {out_data.shape}")
    out_array[block_write] = out_data
    return 0


def downscale_dask(in_array, out_array, factor, write_size, num_workers):
    """
    Downscale the entire volume in parallel using Dask.

    - Partition the output ROI (out_array.shape) into blocks of size write_size.
    - For each output block, compute the corresponding input region as:
         block_read = tuple(slice(s.start * factor[d],
                                  s.start * factor[d] + (s.stop - s.start) * factor[d])
                            for each dimension d)
    - Schedule a delayed task for each block using downscale_block.
    - Use as_completed to report progress and write each block into out_array.
    """
    print("Downsampling by factor %s" % (factor,))
    total_shape = out_array.shape
    blocks, grid_dims = partition_roi(total_shape, write_size)
    print("Processing ROI %s with %d blocks" % (total_shape, len(blocks)))

    scheduled = []
    for block in blocks:
        block_write = block  # region in output dataset
        # Compute corresponding input region using the 'shrink' logic.
        block_read = tuple(slice(s.start * factor[d],
                                 s.start * factor[d] + (s.stop - s.start) * factor[d])
                           for d, s in enumerate(block))
        task = dask.delayed(downscale_block)(in_array, out_array, factor, block_read, block_write)
        scheduled.append((block_write, task))

    client = Client(n_workers=num_workers)
    futures = [client.compute(task) for _, task in scheduled]
    progress_bar = tqdm(total=len(futures), desc="Downscaling Blocks")
    mapping = {f: bw for ((bw, _), f) in zip(scheduled, futures)}
    for future in as_completed(futures):
        future.result()
        progress_bar.update(1)
    progress_bar.close()
    client.close()


def prepare_ds(in_file, ds_name, total_roi, voxel_size, write_size, dtype, num_channels):
    """
    Create a new dataset in the Zarr container with specified ROI, voxel size, and chunk size.

    Returns the newly created zarr.Array with attributes set.
    """
    z = zarr.open(in_file, mode='a')
    offset, shape = total_roi
    ds = z.create_dataset(ds_name, shape=shape, chunks=write_size, dtype=dtype,
                          compressor=Blosc(cname="zlib", clevel=3))
    ds.attrs["voxel_size"] = voxel_size
    ds.attrs["roi"] = total_roi
    ds.attrs["num_channels"] = num_channels
    return ds


def scale_pyramid(in_file, in_ds_name, scales, chunk_shape, num_workers=32):
    """
    Build a multi-resolution pyramid from a raw Zarr dataset.

    Behavior:
      - Verifies that in_ds_name points to a dataset.
      - If in_ds_name does not end with '/s0', renames it accordingly.
      - For each scale, computes:
            next_voxel_size = prev_voxel_size * scale
            next_total_roi = snap_to_grid(prev_roi, next_voxel_size)  [mode: grow]
            next_write_size = chunk_shape * next_voxel_size (element-wise)
        Then creates a new dataset and downscales from the previous resolution.

    A progress bar is shown for each scale.
    """
    ds = zarr.open(in_file, mode='a')

    if in_ds_name not in ds:
        raise RuntimeError("%s does not seem to be a dataset" % in_ds_name)

    # Rename if necessary.
    if not in_ds_name.endswith('/s0'):
        ds_name = in_ds_name + '/s0'
        print("Moving %s to %s" % (in_ds_name, ds_name))
        ds.store.rename(in_ds_name, in_ds_name + '__tmp')
        ds.store.rename(in_ds_name + '__tmp', ds_name)
    else:
        ds_name = in_ds_name
        in_ds_name = in_ds_name[:-3]

    print("Scaling %s by factors %s" % (in_file, scales))

    prev_array = ds[ds_name]

    if chunk_shape is not None:
        chunk_shape = tuple(chunk_shape)
    else:
        chunk_shape = prev_array.chunks
        print("Reusing chunk shape of %s for new datasets" % (chunk_shape,))

    if "num_channels" in prev_array.attrs:
        num_channels = prev_array.attrs["num_channels"]
    else:
        num_channels = 1

    if "voxel_size" in prev_array.attrs:
        prev_voxel_size = tuple(prev_array.attrs["voxel_size"])
    else:
        prev_voxel_size = (1,) * len(prev_array.shape)

    if "roi" in prev_array.attrs:
        prev_roi = tuple(prev_array.attrs["roi"])
    else:
        prev_roi = ((0,) * len(prev_array.shape), prev_array.shape)

    # Outer progress: one bar per scale level.
    for scale_num, scale in enumerate(scales):
        try:
            scale = tuple(scale) if hasattr(scale, '__iter__') else (scale,) * len(prev_voxel_size)
        except Exception:
            scale = (scale,) * len(prev_voxel_size)

        next_voxel_size = tuple(prev_voxel_size[d] * scale[d] for d in range(len(prev_voxel_size)))
        offset, shape = prev_roi
        next_shape = tuple(((shape[d] + next_voxel_size[d] - 1) // next_voxel_size[d]) * next_voxel_size[d]
                           for d in range(len(shape)))
        next_total_roi = (offset, next_shape)
        next_write_size = tuple(chunk_shape[d] * next_voxel_size[d] for d in range(len(chunk_shape)))

        print("Next voxel size: %s" % (next_voxel_size,))
        print("Next total ROI: %s" % (next_total_roi,))
        print("Next chunk size: %s" % (next_write_size,))

        next_ds_name = in_ds_name + '/s' + str(scale_num + 1)
        print("Preparing %s" % (next_ds_name,))

        next_array = prepare_ds(in_file, next_ds_name, next_total_roi, next_voxel_size,
                                next_write_size, prev_array.dtype, num_channels)

        downscale_dask(prev_array, next_array, scale, next_write_size, num_workers=num_workers)

        prev_array = next_array
        prev_voxel_size = next_voxel_size
        prev_roi = next_total_roi


# --- Command-line Interface ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a scale pyramid for a Zarr/N5 container.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--file', '-f',
        type=str,
        required=True,
        help="The input container")
    parser.add_argument(
        '--ds', '-d',
        type=str,
        required=True,
        help="The name of the dataset")
    parser.add_argument(
        '--scales', '-s',
        nargs='*',
        type=int,
        required=True,
        help="The downscaling factors between scales (e.g. 2 4 8)")
    parser.add_argument(
        '--chunk_shape', '-c',
        nargs='*',
        type=int,
        default=None,
        help="The size of a chunk in voxels")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of Dask workers")

    args = parser.parse_args()

    scale_pyramid(args.file, args.ds, args.scales, args.chunk_shape, args.num_workers)
