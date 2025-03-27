#!/usr/bin/env python
"""
Convert 3D zarr arrays to .tif image sequences

This script converts a 3D zarr array (zyx order) into a sequence of TIFF images.
It partitions the volume only along the z-axis so that each block produces a contiguous,
non-overlapping range of sections. Each section is converted to uint8 (using the same rules)
and saved as "section_{z:04d}.tif" in the output directory.

Dependencies:
  - python 3.8+
  - dask[distributed]
  - zarr
  - configargparse
  - numpy
  - scikit-image
  - numcodecs
  - tqdm
"""

import os
import warnings
from time import sleep, time as now

import configargparse as argparse
import numpy as np
import skimage
import skimage.io
import zarr
from dask.distributed import Client, as_completed

# --- Conversion Helpers ---
from skimage.util import img_as_ubyte
from tqdm import tqdm


def convert_to_uint8(array):
    dtype = array.dtype
    if np.dtype(dtype) == np.uint32:
        # For uint32 labels, set non-zero values to 255.
        return (array != 0).astype(np.uint8) * 255
    elif np.issubdtype(dtype, np.integer):
        if array.max() > 255:
            raise ValueError(
                "Array contains integers >255, cannot safely convert to uint8."
            )
        return array.astype(np.uint8)
    elif np.issubdtype(dtype, np.floating):
        if array.min() < 0.0 or array.max() > 1.0:
            raise ValueError(
                "Array contains floats outside [0,1], cannot safely scale to uint8."
            )
        array = img_as_ubyte(array)
        return array.astype(np.uint8)
    else:
        raise TypeError(f"Conversion to uint8 not defined for dtype {dtype}.")


# --- Worker Function ---
def convert_worker(zarr_filename, ds_name, block, out_path):
    """
    Worker function to convert a zarr block to TIFF images.

    Parameters:
      - zarr_filename: path to the zarr container.
      - ds_name: name of the dataset inside the container.
      - block: a tuple (z_slice, full_y, full_x) defining the block ROI.
      - out_path: directory to save TIFF images.

    Behavior:
      - Re-opens the dataset, reads the block, computes the global z indices (using
        ds.attrs "offset" and "voxel_size"), converts each section to uint8, and saves
        each section as "section_{z:04d}.tif".
    """
    # Open the dataset
    ds = zarr.open(os.path.join(zarr_filename, ds_name), mode="r")
    data = np.array(ds[block])

    # Get offset and voxel size; assume they are in attributes or default.
    voxel_size = ds.attrs.get("voxel_size", (1, 1, 1))
    offset = ds.attrs.get("offset", (0, 0, 0))
    ds_offset_z = offset[0] / voxel_size[0]

    # Compute the global starting z-index for this block.
    # Here block[0] is a slice for z.
    start_z = int(block[0].start / voxel_size[0] - ds_offset_z)
    stop_z = int(block[0].stop / voxel_size[0] - ds_offset_z)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for section, z in zip(data, range(start_z, stop_z)):
            section = convert_to_uint8(section)
            filename_out = os.path.join(out_path, f"section_{z:04d}.tif")
            skimage.io.imsave(filename_out, section)
    return 0


# --- Partitioning Function (Z-axis only) ---
def partition_z_axis(total_shape, z_chunk):
    """
    Partition a 3D volume (zyx) along the z-axis only.

    Returns a list of blocks where each block is a tuple:
      (slice(z0, z1), slice(0, Y), slice(0, X))
    """
    z, y, x = total_shape
    blocks = []
    n_z = (z + z_chunk - 1) // z_chunk
    for i in range(n_z):
        z0 = i * z_chunk
        z1 = min((i + 1) * z_chunk, z)
        blocks.append((slice(z0, z1), slice(0, y), slice(0, x)))
    return blocks


# --- Main Conversion Function ---
def convert(filename, ds_name, out_path, num_workers):
    """
    Convert a 3D zarr array to a sequence of TIFF images.

    - Opens the zarr dataset.
    - Ensures it is 3D.
    - Partitions the volume along the z-axis using the dataset's chunk size for z.
    - Submits a task for each block using dask.distributed.Client.
    - Each block's worker writes its TIFF files to out_path.
    """
    logger = __import__("logging").getLogger(__name__)
    logger.info(f"Converting {os.path.join(filename, ds_name)}")
    start = now()

    # Open the dataset and get its shape and chunking.
    ds_path = os.path.join(filename, ds_name)
    ds = zarr.open(ds_path, mode="r")
    shape = ds.shape
    if len(shape) != 3:
        raise NotImplementedError("Conversion only implemented for 3D zarr arrays")

    # Warn if dtype is not uint8.
    if np.dtype(ds.dtype) != np.uint8:  # type: ignore
        logger.warning(f"Input dtype {ds.dtype} does not match output dtype uint8.")

    # For ordering, partition only along z. Use the first element of the chunk shape.
    z_chunk = ds.chunks[0]
    blocks = partition_z_axis(shape, z_chunk)
    total_blocks = len(blocks)
    logger.info(
        f"Processing volume of shape {shape} in {total_blocks} blocks (z-axis partition)"
    )

    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    client = Client(n_workers=num_workers)
    logger.info(f"Dask client created with {num_workers} workers")
    sleep(2)

    scheduled = []
    for block in blocks:
        future = client.submit(convert_worker, filename, ds_name, block, out_path)
        scheduled.append((block, future))

    progress_bar = tqdm(total=len(scheduled), desc="Processing Blocks")
    mapping = {f: block for block, f in scheduled}
    for future in as_completed([f for _, f in scheduled]):
        mapping[future]  # retrieve block (unused here)
        future.result()
        progress_bar.update(1)
    progress_bar.close()

    client.close()
    logger.info(f"Done in {now() - start:.2f} s")


# --- Command-line Interface (if needed) ---
if __name__ == "__main__":
    import configargparse as argparse

    parser = argparse.ArgParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--filename", "-f", required=True, help="Path to the zarr container"
    )
    parser.add_argument(
        "--dataset", "-d", required=True, help="Name of the dataset in the container"
    )
    parser.add_argument(
        "--out_path", "-o", required=True, help="Output directory for TIFF images"
    )
    parser.add_argument(
        "--num_workers", "-n", type=int, default=16, help="Number of Dask workers"
    )
    args = parser.parse_args()
    convert(args.filename, args.dataset, args.out_path, args.num_workers)
