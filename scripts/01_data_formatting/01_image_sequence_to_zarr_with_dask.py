"""
Store a sequence of 2D .tif images in 3D blockwise zarr format.

This script does not run in the default incasem environmet due to dependency
conflicts between dask and daisy.
Please create a new environment and install the following dependencies:
- python 3.8
- dask[complete]
- zarr
- configargparse
- scikit-image
- tqdm
"""

import logging
import os
import os.path as osp
import re
from pathlib import Path
from glob import glob

import numpy as np

import dask
from dask import delayed
from dask.diagnostics import ProgressBar
from dask.array.image import imread as lazy_imread
import zarr
from numcodecs import Blosc
import configargparse as argparse
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def check_image_paths(raw_dir, regex):
    file_names = [
        f for f in os.listdir(raw_dir) if osp.isfile(
            osp.join(raw_dir, f))
    ]

    image_paths = sorted([re.search(regex, f).group(0)
                          for f in file_names])
    logger.debug(f'first 20 image paths {image_paths[:20]}')

    # check for missing images
    section_numbers = sorted([int(re.search(regex, f).group(1))
                              for f in file_names])

    assert section_numbers == list(range(
        section_numbers[0], section_numbers[-1] + 1)), \
        'There is a problem with section numbering'

    image_paths = [osp.join(raw_dir, f) for f in image_paths]
    return image_paths


def verify_images(globstring):
    for p in tqdm(glob(globstring), desc="Verify images"):
        try:
            img = Image.open(str(p))
            img.verify()
        except BaseException:
            raise RuntimeError(f"{p}")


def image_sequence_to_zarr(
        raw_dir,
        output_file,
        output_dataset,
        image_regex=r'.*_(\d+).*\.tif$',
        dtype=np.uint8,
        resolution=(1, 1, 1),
        chunks=(128, 128, 128),
):
    """Store a sequence of 2D .tif images in 3D blockwise zarr format.

    Args:
        raw_dir (str):

            Directory with 2D section images

        output_file (str):

            Zarr file (zarr.Group) for output

        output_dataset (str):

            Zarr dataset (zarr.Array) for output

        image_regex (str):

            Regex to select sections and extract their numbers

        dtype (np.dtype):

            Datatype of output

        resolution (tuple of int):

            z, y, x resolution in nanometers

        chunks (tuple of int):

            z, y, x chunk size of output blocks in voxels
    """
    _ = check_image_paths(raw_dir, image_regex)

    path = Path(raw_dir) / "*.tif"
    verify_images(str(path))
    stack = lazy_imread(str(path))

    input_dt = stack.dtype
    output_dt = np.dtype(dtype)
    if input_dt is not output_dt:
        logger.warning(
            f'output dtype {output_dt} does not match input dtype {input_dt}')

    blocks = stack.rechunk((chunks[0],) + stack.shape[1:])

    z = zarr.open(
        store=output_file,
        path=output_dataset,
        mode="w-",
        shape=stack.shape,
        chunks=chunks,
        dtype=output_dt,
        compressor=Blosc(cname="zlib", clevel=3),
    )
    z.attrs.put({"offset": [0, 0, 0], "resolution": list(resolution)})
    stored = dask.array.to_zarr(blocks, z, compute=False)

    # dask.visualize(stored, filename="dask_graph.png")

    with ProgressBar():
        dask.compute(stored)


def parse_args():
    p = argparse.ArgParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        usage="""
        Store a sequence of 2D .tif images in 3D blockwise zarr format.

        This script does not run in the default incasem environmet due to dependency
        conflicts between dask and daisy.
        Please create a new environment and install the following dependencies:
        - python 3.8
        - dask[complete]
        - zarr
        - configargparse
        - scikit-image
        - tqdm
        """
    )
    p.add(
        '-c',
        '--config',
        is_config_file=True,
        help='config file path')
    p.add(
        '-i',
        '--input_dir',
        nargs='+',
        required=True,
        help='Directions with section images.')
    p.add(
        '-f',
        '--output_file',
        required=True,
        help='Zarr file to be created.')
    p.add(
        '-d',
        '--output_dataset',
        required=True,
        nargs='+',
        help='Datasets inside zarr file.')
    p.add(
        '-r',
        '--image_regex',
        default=r'.*_(\d+).*\.tif$',
        help='Regex to select sections and extract their numbers.')
    p.add(
        '--dtype',
        nargs='+',
        default=['uint8'],
        help='Any numpy datatype.')
    p.add(
        '--offset',
        default=[0, 0, 0],
        type=int,
        nargs=3,
        help='z,y,x offset in voxels.')
    p.add(
        '--resolution',
        required=True,
        type=int,
        nargs=3,
        help='z,y,x resolution in nanometers.')
    p.add(
        '--chunks',
        default=[128, 128, 128],
        type=int,
        nargs=3,
        help='z,y,x chunk size in voxels.')

    args = p.parse_args()
    logger.info(f'\n{p.format_values()}')

    return args


if __name__ == '__main__':
    args = parse_args()

    for i in range(len(args.input_dir)):
        logger.info((
            f'Converting from {args.input_dir[i]} '
            f'to {args.output_dataset[i]}...'
        ))
        image_sequence_to_zarr(
            args.input_dir[i],
            args.output_file,
            args.output_dataset[i],
            args.image_regex,
            args.dtype[i],
            args.resolution,
            args.chunks,
        )
