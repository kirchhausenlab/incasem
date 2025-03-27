#!/usr/bin/env python
"""Create a mask to ignore predictions at the boundary of objects using Dask.

This script replicates the original behavior exactly by dividing the input labels
volume (stored in a Zarr container) into blocks of a specified chunk size, growing
each block by a context (set by the maximum of the exclude parameters), applying
binary dilation and erosion to compute a boundary mask, and then writing the
inverted mask (0/255) into a new dataset in the same container.

Dependencies:
  - python 3.8+
  - numpy
  - configargparse
  - scipy
  - dask[distributed]
  - dask
  - zarr
  - numcodecs
  - tqdm
"""

import logging
from time import time as now, sleep
import itertools
import numpy as np
import configargparse as argparse
from scipy import ndimage
import dask
from dask.distributed import Client, as_completed
import zarr
from numcodecs import Blosc
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_metric_block(
    core_slice,
    context,
    filename,
    ds_name,
    exclude_voxels_outwards,
    exclude_voxels_inwards,
):
    """
    Process one block of the labels volume.

    Opens the input Zarr file (to avoid pickling non-serializable objects), reads
    a region defined as the core (write) region grown by a uniform context, pads if
    necessary (using 0), binarizes the data, applies dilation and erosion to compute
    a boundary mask, inverts it (0/255), and returns only the core region.

    Parameters
    ----------
    core_slice : tuple of slices
        Slices defining the core (write) region in global coordinates.
    context : int
        Number of voxels to add as context on each side.
    filename : str
        Path to the Zarr container.
    ds_name : str
        Name of the input labels dataset.
    exclude_voxels_outwards : int
        Number of dilation iterations.
    exclude_voxels_inwards : int
        Number of erosion iterations.

    Returns
    -------
    np.ndarray (uint8)
        Processed mask for the core region.
    """
    # Re-open the Zarr file in read-only mode.
    z = zarr.open(filename, mode="r")
    labels = z[ds_name]
    global_shape = labels.shape
    ndim = len(global_shape)

    # Compute the read region: extend each core_slice by 'context' on both sides.
    read_slices = []
    pad_width = []
    for d in range(ndim):
        rstart = core_slice[d].start - context
        rend = core_slice[d].stop + context
        actual_start = max(rstart, 0)
        actual_end = min(rend, global_shape[d])
        read_slices.append(slice(actual_start, actual_end))
        pad_left = 0 if rstart >= 0 else -rstart
        pad_right = 0 if rend <= global_shape[d] else rend - global_shape[d]
        pad_width.append((pad_left, pad_right))

    # Load the read region and pad with 0 if necessary.
    data = labels[tuple(read_slices)]
    data = np.array(data)
    if any(p != (0, 0) for p in pad_width):
        data = np.pad(data, pad_width, mode="constant", constant_values=0)

    # Binarize: mark nonzero as foreground.
    data = (data != 0).astype(np.uint8)

    # Apply dilation and erosion.
    dilated = ndimage.binary_dilation(data, iterations=exclude_voxels_outwards).astype(
        np.uint8
    )
    eroded = ndimage.binary_erosion(data, iterations=exclude_voxels_inwards).astype(
        np.uint8
    )
    boundary_mask = dilated - eroded

    # Invert boundary mask: foreground becomes 0 and background 255.
    mask = np.logical_not(boundary_mask).astype(np.uint8) * 255

    # Extract the core region from the processed extended block.
    core_slices = []
    for d in range(ndim):
        rstart = core_slice[d].start - context
        offset = core_slice[d].start - (rstart if rstart >= 0 else 0)
        length = core_slice[d].stop - core_slice[d].start
        core_slices.append(slice(offset, offset + length))
    result = mask[tuple(core_slices)]
    return result


def create_metric_mask(
    filename,
    ds_name,
    out_ds_name,
    chunk_shape,
    exclude_voxels_outwards,
    exclude_voxels_inwards,
    num_workers,
):
    """
    Create the metric mask over the entire volume using per-block processing.

    The input labels volume is partitioned into blocks of size `chunk_shape`. Each block
    is grown by a uniform context (equal to max(exclude_voxels_outwards, exclude_voxels_inwards))
    to ensure correct boundary handling. Each block is processed individually and stored
    directly into the output Zarr dataset. A tqdm progress bar reports block-by-block progress.
    """
    # Open input dataset and determine global shape.
    zarr_file = zarr.open(filename, mode="r")
    labels = zarr_file[ds_name]
    global_shape = labels.shape
    ndim = len(global_shape)

    # For simplicity, assume voxel size is 1; context is then just:
    context = max(exclude_voxels_outwards, exclude_voxels_inwards)
    logger.info("Using context (voxels): %d", context)

    # Compute grid dimensions.
    grid_dims = [
        (global_shape[d] + chunk_shape[d] - 1) // chunk_shape[d] for d in range(ndim)
    ]
    total_blocks = np.prod(grid_dims)
    logger.info(
        "Processing volume of shape %s in grid %s (%d blocks)",
        global_shape,
        grid_dims,
        total_blocks,
    )

    # Create a Dask client.
    client = Client(n_workers=num_workers)
    logger.info("Dask client created with %d workers", num_workers)
    logger.info("Dask dashboard available at %s", client.dashboard_link)
    sleep(2)
    logger.info("Dask workers: %s", client.nthreads())

    # Open (or create) the output dataset in the same Zarr container.
    out_zarr = zarr.open(filename, mode="a")
    if out_ds_name in out_zarr:
        logger.info("Deleting existing dataset '%s' in %s", out_ds_name, filename)
        del out_zarr[out_ds_name]
    out_array = out_zarr.create_dataset(
        name=out_ds_name,
        shape=global_shape,
        chunks=tuple(chunk_shape),
        dtype=np.uint8,
        compressor=Blosc(cname="zlib", clevel=3),
    )

    # Schedule each block.
    scheduled = []
    for idx in tqdm(
        itertools.product(*(range(n) for n in grid_dims)), total=total_blocks
    ):
        core_slice = tuple(
            slice(
                idx[d] * chunk_shape[d],
                min((idx[d] + 1) * chunk_shape[d], global_shape[d]),
            )
            for d in range(ndim)
        )
        future = client.compute(
            dask.delayed(process_metric_block)(
                core_slice,
                context,
                filename,
                ds_name,
                exclude_voxels_outwards,
                exclude_voxels_inwards,
            )
        )
        scheduled.append((idx, core_slice, future))

    logger.info("Processing %d blocks...", len(scheduled))
    # Create a mapping from future to (index, core_slice)
    mapping = {entry[2]: (entry[0], entry[1]) for entry in scheduled}
    futures = [entry[2] for entry in scheduled]

    progress_bar = tqdm(total=len(futures), desc="Processing Blocks")
    # Use as_completed to iterate over futures as they finish.
    for future in as_completed(futures):
        idx, core_slice = mapping[future]
        block_result = future.result()  # Wait for the block result.
        out_array[core_slice] = block_result  # Write block to its global position.
        progress_bar.update(1)
    progress_bar.close()

    logger.info("All blocks processed and stored.")
    client.close()


def parse_args():
    p = argparse.ArgParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add("--config", is_config_file=True, help="Config file path")
    p.add("--filename", "-f", required=True, help="Input Zarr container filename")
    p.add("--dataset", "-d", required=True, help="Labels dataset to use for masking")
    p.add(
        "--out_dataset", "-o", required=True, help="Name of the new metric mask dataset"
    )
    p.add(
        "--chunk_shape",
        "-c",
        nargs="+",
        type=int,
        default=[128, 128, 128],
        help="Size of an output chunk in voxels",
    )
    p.add(
        "--exclude_voxels_outwards",
        type=int,
        default=4,
        help="Number of voxels to dilate (outwards)",
    )
    p.add(
        "--exclude_voxels_inwards",
        type=int,
        default=4,
        help="Number of voxels to erode (inwards)",
    )
    p.add("--num_workers", "-n", type=int, default=32, help="Number of Dask workers")
    args = p.parse_args()
    logger.info("\n%s", p.format_values())
    return args


def main():
    args = parse_args()
    create_metric_mask(
        args.filename,
        args.dataset,
        args.out_dataset,
        args.chunk_shape,
        args.exclude_voxels_outwards,
        args.exclude_voxels_inwards,
        args.num_workers,
    )


if __name__ == "__main__":
    main()
