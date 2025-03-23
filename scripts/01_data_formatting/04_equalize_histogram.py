#!/usr/bin/env python
"""Apply contrast limited adaptive histogram equalization using dask.

This script replicates the original behavior (using funlib and daisy) by dividing
the raw Zarr volume into blocks, “growing” each block by a context determined by the
CLAHE kernel size, and then applying CLAHE with the same parameters. Blocks at the
image boundaries are padded using the mean of the core region, exactly as before.

Dependencies:
  - python 3.8+
  - numpy
  - configargparse
  - dask[distributed]
  - dask
  - zarr
  - numcodecs
  - scikit-image
  - incasem.utils (for equalize_adapthist)
"""

import logging
from time import time as now, sleep
import itertools
import numpy as np
import configargparse as argparse
import dask
from dask.distributed import Client, as_completed
import zarr
from numcodecs import Blosc
from tqdm import tqdm

# Import the CLAHE function (algorithm identical to the original)
from incasem.utils import equalize_adapthist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_block_delayed(core_slice, context, raw, kernel_size, clip_limit):
    """
    Process one block of the raw volume.

    Parameters
    ----------
    core_slice : tuple of slices
        The core (write) region for this block in global coordinates.
    context : tuple of int
        The context (overlap) to add on each side.
        (Computed originally as: (kernel_size + 1) // 2 per axis.)
    raw : zarr.Array
        The raw dataset.
    kernel_size : int
        CLAHE kernel size.
    clip_limit : float
        CLAHE clip limit.

    Returns
    -------
    np.ndarray (uint8)
        The processed (CLAHE-equalized) core region.
    """
    global_shape = raw.shape
    ndim = len(global_shape)

    # Compute read region: extend core by context on each side.
    read_starts = []
    read_ends = []
    for d in range(ndim):
        core_start = core_slice[d].start
        core_end = core_slice[d].stop
        rstart = core_start - context[d]
        rend = core_end + context[d]
        read_starts.append(rstart)
        read_ends.append(rend)

    # Determine available slices and required padding.
    actual_slices = []
    pad_width = []
    for d in range(ndim):
        rstart = read_starts[d]
        rend = read_ends[d]
        actual_start = max(rstart, 0)
        actual_end = min(rend, global_shape[d])
        actual_slices.append(slice(actual_start, actual_end))
        pad_left = 0 if rstart >= 0 else -rstart
        pad_right = 0 if rend <= global_shape[d] else rend - global_shape[d]
        pad_width.append((pad_left, pad_right))

    # Compute mean of the core region.
    core_data = raw[core_slice]
    mean_val = int(np.mean(core_data))

    # Extract the read region and pad if needed.
    read_data = raw[tuple(actual_slices)]
    read_data = np.array(read_data)
    if any(p != (0, 0) for p in pad_width):
        read_data = np.pad(read_data, pad_width,
                           mode='constant', constant_values=mean_val)

    # Apply CLAHE.
    equalized = equalize_adapthist(
        image=read_data,
        kernel_size=kernel_size,
        clip_limit=clip_limit,
        nbins=256
    )
    if equalized.dtype != np.uint8:
        equalized = equalized.astype(np.uint8)

    # Extract the core region from the equalized result.
    core_slices = []
    for d in range(ndim):
        offset = core_slice[d].start - read_starts[d]
        length = core_slice[d].stop - core_slice[d].start
        core_slices.append(slice(offset, offset + length))
    result = equalized[tuple(core_slices)]
    return result


def equalize_histogram(filename, ds_name, out_ds_name, chunk_shape, kernel_size, clip_limit, num_workers):
    """
    Apply CLAHE over the whole volume in blocks.

    Instead of building one huge Dask graph, this version computes each block
    individually and writes it directly into the output Zarr dataset. Progress
    is reported block by block.
    """
    # Open raw dataset (read-only)
    zarr_file = zarr.open(filename, mode='r')
    raw = zarr_file[ds_name]
    global_shape = raw.shape
    ndim = len(global_shape)
    context = tuple(int((kernel_size + 1) // 2) for _ in range(ndim))
    grid_dims = [(global_shape[d] + chunk_shape[d] - 1) // chunk_shape[d]
                 for d in range(ndim)]
    total_blocks = np.prod(grid_dims)

    # Create a Dask client.
    client = Client(n_workers=num_workers)
    logger.info("Dask client created with %d workers", num_workers)
    logger.info("Dask dashboard available at %s", client.dashboard_link)
    sleep(2)
    logger.info("Dask workers: %s", client.nthreads())
    logger.info("Processing volume with shape %s in grid %s", global_shape, grid_dims)

    # Open (or create) output dataset in the same Zarr container.
    out_zarr = zarr.open(filename, mode='a')
    if out_ds_name in out_zarr:
        logger.info("Deleting existing dataset '%s' in %s", out_ds_name, filename)
        del out_zarr[out_ds_name]
    out_array = out_zarr.create_dataset(
        name=out_ds_name,
        shape=global_shape,
        chunks=tuple(chunk_shape),
        dtype=np.uint8,
        compressor=Blosc(cname="zlib", clevel=3)
    )

    # Schedule and compute each block individually.
    scheduled = []
    for idx in itertools.product(*(range(n) for n in grid_dims)):
        core_slice = tuple(
            slice(idx[d] * chunk_shape[d], min((idx[d] + 1) * chunk_shape[d], global_shape[d]))
            for d in range(ndim)
        )
        delayed_block = dask.delayed(process_block_delayed)(core_slice, context, raw, kernel_size, clip_limit)
        scheduled.append((idx, core_slice, client.compute(delayed_block)))

    logger.info("Processing %d blocks...", len(scheduled))
    progress_bar = tqdm(total=len(scheduled), desc="Processing Blocks")
    # Process each block as it becomes available.
    for idx, core_slice, future in scheduled:
        block_result = future.result()  # Blocks until this block is computed.
        out_array[core_slice] = block_result  # Write the block to its global position.
        progress_bar.update(1)
    progress_bar.close()

    logger.info("All blocks processed and stored.")
    client.close()


def parse_args():
    p = argparse.ArgParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add('--config', is_config_file=True, help='Config file path.')
    p.add('--filename', '-f', required=True, help='Input Zarr container filename.')
    p.add('--dataset', '-d', default='volumes/raw', help='Name of the raw dataset.')
    p.add('--out_dataset', '-o', default='volumes/raw_equalized_0.02', help='Name of the new output dataset.')
    p.add('--chunk_shape', '-c', nargs='+', type=int, default=[128, 128, 128], help='Size of a chunk in voxels.')
    p.add('--kernel_size', type=int, default=128, help='Block edge length for CLAHE.')
    p.add('--clip_limit', type=float, default=0.02, help='Clip relative frequency for adapting the histogram.')
    p.add('--num_workers', '-n', type=int, default=20, help='Number of dask workers.')
    args = p.parse_args()
    logger.info("\n%s", p.format_values())
    return args


def main():
    args = parse_args()
    equalize_histogram(
        args.filename,
        args.dataset,
        args.out_dataset,
        args.chunk_shape,
        args.kernel_size,
        args.clip_limit,
        args.num_workers
    )


if __name__ == '__main__':
    main()
