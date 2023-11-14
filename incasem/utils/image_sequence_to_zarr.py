'''Store a sequence of 2D images in blockwise zarr format'''

import os
import os.path as osp
import logging
import re
from time import time as now

import numpy as np
import PIL
from PIL import Image

from funlib.persistence import Array, open_ds, prepare_ds
from funlib.geometry import Roi, Coordinate
import daisy

from .image_conversions import *

# Allow for large images that would throw a DecompressionBombError
PIL.Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_image_paths(raw_dir, regex):
    ''' describe '''
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


def determine_roi(image_paths, offset=(0, 0, 0), shape=(None, None, None)):
    """Determine daisy.Roi from parameters and present data

    Args:
        image_paths (List of str):

            Absolute paths to all images

        offset (tuple of int):

            z, y, x offset to extract from present data

        shape:

            z, y, x shape to extract from present data

    Returns:
        daisy.Roi
    """

    offset = Coordinate(offset)

    # get image x,y sizes
    img = Image.open(image_paths[0])
    img_array = np.array(img)

    # slice_array.shape is y,x here
    z_shape = len(image_paths)
    if shape[0]:
        assert offset[0] + shape[0] <= z_shape, \
            'desired z shape exceeds data shape'
        z_shape = shape[0]

    y_shape = img_array.shape[0]
    if shape[1]:
        assert offset[1] + shape[1] <= y_shape, \
            'desired y shape exceeds data shape'
        y_shape = shape[1]

    x_shape = img_array.shape[1]
    if shape[2]:
        assert offset[2] + shape[2] <= x_shape, \
            'desired x shape exceeds data shape'
        x_shape = shape[2]

    shape_cropped = Coordinate([z_shape, y_shape, x_shape])

    roi = Roi(
        offset,
        shape_cropped
    )

    return roi


def get_images_dtype(image_path):
    """get_images_dtype.

    Args:
        image_path:
    """
    img = Image.open(image_path)
    img = img.crop((0, 0, 1, 1))
    array = np.array(img)
    return array.dtype


def write_to_zarr(
        block,
        voxel_size,
        image_paths,
        dataset,
        dtype,
        conversion=None,
        invert=False):
    """Write a daisy.Block to persistent zarr array

    Args:
        block (daisy.Block):
        voxel_size (daisy.Coordinate):
        image_paths (List of str):
        dataset (daisy.Dataset):
        dtype (np.dtype):
        conversion (ImageConversion):
        invert (bool)
    """

    imgs = []

    offset_index = block.read_roi.get_offset() / voxel_size
    end_index = block.read_roi.get_end() / voxel_size

    paths_to_load = image_paths[offset_index[0]:end_index[0]]
    logger.debug(f'paths to load {paths_to_load}')

    for idx, i in enumerate(paths_to_load):
        # logger.debug(f'loading image number {idx} ...')
        img = Image.open(i)
        # lazily crop the slice. img has shape x, y
        img = img.crop((
            offset_index[2],
            offset_index[1],
            end_index[2],
            end_index[1]
        ))
        if conversion:
            img = conversion(img)
        # array has shape y, x
        array = np.array(img)
        imgs.append(array)

    # if individual images have not been converted to output dtype, conversion
    # happens here
    stack = np.array(imgs, dtype=dtype)
    assert np.all(stack.shape == block.read_roi.get_shape() / voxel_size), \
        'Actual stack shape does not match ROI definition'
    logger.debug(f'stack shape (z, y, x): {stack.shape}')

    if invert:
        assert np.max(stack) < 256
        stack = 255 - stack

    dataset[block.write_roi] = stack


def image_sequence_to_zarr(
        raw_dir,
        output_file,
        output_dataset,
        image_regex,
        dtype,
        offset,
        shape,
        resolution,
        chunks,
        conversion,
        num_workers,
        invert=False):
    """Store a sequence of 2D images in blockwise zarr format

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

        offset (tuple of int):

            z, y, x offset in voxels within present data

        shape (tuple of int):

            z, y, x shape in voxels within present data

        resolution (tuple of int):

            z, y, x resolution in nanometers

        chunks (tuple of int):

            z, y, x chunk size of output blocks in voxels

        conversion (ImageConversion):

            conversion of each 2D slice, using Pillow and Numpy

        num_workers (int):

            number of CPU cores for blockwise processing

        invert (bool):

            if True, invert LUT of input .tiff files

    """

    image_paths = get_image_paths(raw_dir, image_regex)
    roi = determine_roi(image_paths, offset, shape)
    voxel_size = Coordinate(resolution)
    metric_roi = roi * voxel_size
    if conversion:
        conversion = globals()[conversion](
            dtype=dtype,
            input_path=osp.abspath(raw_dir)
        )

    input_dt = get_images_dtype(image_paths[0])
    output_dt = np.dtype(dtype)
    if input_dt is not output_dt:
        logger.warning(
            f'output dtype {output_dt} does not match input dtype {input_dt}')

    ds = prepare_ds(output_file,
                          output_dataset,
                          total_roi=metric_roi,
                          voxel_size=voxel_size,
                          write_size=voxel_size * Coordinate(chunks),
                          dtype=dtype,
                          compressor={'id': 'zlib', 'level': 3})

    start = now()
    # Spawn a worker per chunk
    block_roi = Roi(
        (0, 0, 0),
        voxel_size * Coordinate(chunks)
    )

    task = daisy.Task(
        total_roi=metric_roi,
        read_roi=block_roi,
        write_roi=block_roi,
        process_function=lambda block: write_to_zarr(
            block,
            voxel_size,
            image_paths,
            ds,
            dtype,
            conversion,
            invert=invert
        ),
        fit='shrink',
        read_write_conflict=False,
        num_workers=num_workers,
        task_id="image_sequence_to_zarr"
    )

    daisy.run_blockwise([task])

    logger.info(f'Saved to persistent zarr array in {now() - start} s')