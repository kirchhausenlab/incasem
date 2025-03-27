#!/usr/bin/env python
"""
Crop a dataset from a Zarr container.

This module provides functions to crop (or pad) a dataset from a Zarr file.
It replicates the behavior of the original implementation using Dask for parallel
block‐wise processing, without any dependencies on funlib or daisy.

Functions:
  - crop_dataset_worker(...): crops one block.
  - crop_dataset(...): crops one dataset.
  - crop_datasets(...): crops multiple datasets (remains callable).
"""

import itertools
import logging
from time import sleep, time as now

import zarr
from dask.distributed import Client, as_completed
from numcodecs import Blosc
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# --- ROI Helper Functions ---
def add_shift_to_roi(roi, shift):
    """
    Given an ROI represented as a tuple of slices and a shift (tuple of ints),
    return a new ROI with each slice shifted.
    """
    return tuple(
        slice(s.start + shift[i], s.stop + shift[i]) for i, s in enumerate(roi)
    )


def roi_shape(roi):
    """Return the shape (tuple of ints) corresponding to a tuple of slices."""
    return tuple(s.stop - s.start for s in roi)


# --- Minimal Dataset Creation (replacement for prepare_ds) ---
def prepare_new_dataset(
    zarr_filename, ds_name, total_roi, voxel_size, write_size, dtype, compressor
):
    """
    Create a new dataset in the given Zarr container.

    total_roi: a tuple (offset, shape) where shape is used as the dataset shape.
    voxel_size is stored in the attributes.
    write_size: chunk size.
    """
    container = zarr.open(zarr_filename, mode="a")
    offset, shape = total_roi
    ds = container.create_dataset(  # type: ignore
        ds_name,
        shape=shape,
        chunks=write_size,
        dtype=dtype,
        compressor=compressor,
        overwrite=True,
    )
    ds.attrs["voxel_size"] = voxel_size
    ds.attrs["roi"] = total_roi
    return ds


# --- Open Dataset Helper ---
def open_dataset(zarr_filename, ds_name, mode="r"):
    container = zarr.open(zarr_filename, mode=mode)
    return container[ds_name]


# --- Minimal Block Object ---
class Block:
    def __init__(self, roi):
        self.read_roi = roi  # tuple of slices
        self.write_roi = roi


# --- Partition ROI into Blocks ---
def partition_roi(total_shape, block_shape):
    """
    Partition a volume (given as a tuple total_shape) into blocks of size block_shape.
    Implements a "shrink" fit: the last block in each dimension is shrunk to fit.

    Returns:
      - A list of block ROIs (each as a tuple of slices).
      - Grid dimensions as a tuple.
    """
    dims = len(total_shape)
    grid_dims = [
        (total_shape[d] + block_shape[d] - 1) // block_shape[d] for d in range(dims)
    ]
    blocks = []
    for idx in itertools.product(*(range(n) for n in grid_dims)):
        block = tuple(
            slice(
                idx[d] * block_shape[d],
                min((idx[d] + 1) * block_shape[d], total_shape[d]),
            )
            for d in range(dims)
        )
        blocks.append(block)
    return blocks, tuple(grid_dims)


# --- Worker Function ---
def crop_dataset_worker(block, in_ds, out_ds, read_shift):
    """
    Worker function to crop a block from the input dataset and write it to the output dataset.

    in_ds and out_ds are zarr arrays.
    block: a Block object whose read_roi and write_roi are tuples of slices.
    read_shift: a tuple of ints to add to each slice in the read_roi (to account for cropping offset).
    """
    shifted_roi = add_shift_to_roi(block.read_roi, read_shift)
    data = in_ds[shifted_roi]
    # Write to output dataset: assume the output region matches the shape of block.write_roi.
    out_ds[block.write_roi] = data[
        tuple(slice(0, d) for d in roi_shape(block.write_roi))
    ]
    return 0


# --- Main Function: crop_dataset ---
def crop_dataset(
    zarr_filename,
    ds_name,
    out_filename,
    out_ds_name,
    crop_offset_voxels,
    crop_shape_voxels,
    chunk_shape,
    dtype,
    num_workers,
):
    """
    Crop a dataset from a Zarr container.

    Args:
      zarr_filename: Source Zarr container path.
      ds_name: Name of the source dataset.
      out_filename: Output Zarr container path.
      out_ds_name: Name for the output dataset.
      crop_offset_voxels: Desired offset (tuple of ints) or None.
      crop_shape_voxels: Desired shape (tuple of ints) or None.
      chunk_shape: Output block (chunk) shape (tuple of ints).
      dtype: Output numpy dtype.
      num_workers: Number of Dask workers.
    """
    in_ds = open_dataset(zarr_filename, ds_name, mode="r")
    # Assume input dataset has attributes "roi" and "voxel_size"; otherwise default.
    full_roi = in_ds.attrs.get("roi", ((0,) * len(in_ds.shape), in_ds.shape))  # type: ignore
    voxel_size = in_ds.attrs.get("voxel_size", (1,) * len(in_ds.shape))  # type: ignore

    if crop_offset_voxels is None:
        crop_offset = full_roi[0]
    else:
        crop_offset = tuple(crop_offset_voxels)

    if crop_shape_voxels is None:
        crop_shape = full_roi[1]
    else:
        crop_shape = tuple(crop_shape_voxels)

    roi_to_copy = (crop_offset, crop_shape)
    logger.debug(f"Initial ROI to copy: {roi_to_copy}")

    # Shift ROI to zero origin.
    shift_to_origin = tuple(min(x, 0) for x in crop_offset)
    shifted_roi = (
        tuple(crop_offset[i] - shift_to_origin[i] for i in range(len(crop_offset))),
        crop_shape,
    )
    logger.debug(f"Shifted ROI: {shifted_roi}")

    out_ds = prepare_new_dataset(
        out_filename,
        out_ds_name,
        shifted_roi,
        voxel_size,
        tuple(chunk_shape),
        dtype,
        Blosc(cname="zlib", clevel=3),
    )

    total_out_shape = shifted_roi[1]
    blocks_slices, _ = partition_roi(total_out_shape, tuple(chunk_shape))
    blocks = [Block(roi=s) for s in blocks_slices]

    start = now()
    client = Client(n_workers=num_workers)
    logger.info(f"Dask client created with {num_workers} workers")
    sleep(2)

    scheduled = []
    for block in blocks:
        scheduled.append((
            block,
            client.submit(crop_dataset_worker, block, in_ds, out_ds, shift_to_origin),
        ))

    progress_bar = tqdm(total=len(scheduled), desc="Processing Blocks")
    mapping = {f: block for block, f in scheduled}
    for future in as_completed([f for _, f in scheduled]):
        mapping[future]
        future.result()
        progress_bar.update(1)
    progress_bar.close()

    client.close()
    logger.info(f"Done with {out_ds_name} in {now() - start:.2f} s")


# --- Top-level Function: crop_datasets ---
def crop_datasets(
    zarr_filename,
    out_filename,
    datasets,
    out_datasets,
    offset_voxels,
    shape_voxels,
    chunk_shape,
    dtypes,
    num_workers,
):
    """
    Crop or pad multiple datasets in a Zarr container.

    Args:
      zarr_filename (str): Source Zarr container path.
      out_filename (str): Output Zarr container path.
      datasets (sequence of str): Names of source datasets.
      out_datasets (sequence of str): Names for output datasets.
      offset_voxels (tuple of ints or None): Desired offset (zyx) for all outputs.
      shape_voxels (tuple of ints or None): Desired shape (zyx) for all outputs.
      chunk_shape (tuple of ints): Output chunk shape (zyx).
      dtypes (sequence of str): Numpy dtypes for outputs.
      num_workers (int): Number of Dask workers.
    """
    assert len(datasets) == len(out_datasets), (
        "Provide one out_dataset for each dataset."
    )
    assert len(datasets) == len(dtypes), "Provide one dtype for each dataset."

    for ds, out_ds, dt in zip(datasets, out_datasets, dtypes):
        crop_dataset(
            zarr_filename,
            ds,
            out_filename,
            out_ds,
            offset_voxels,
            shape_voxels,
            chunk_shape,
            dt,
            num_workers,
        )
