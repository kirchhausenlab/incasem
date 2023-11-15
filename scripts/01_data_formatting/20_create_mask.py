'''Create a foreground mask for raw EM Data'''

import logging
from time import time as now

import numpy as np
import configargparse as argparse
import skimage
from skimage.morphology import ball

from funlib.persistence import Array, open_ds, prepare_ds
from funlib.geometry import Roi, Coordinate
import daisy


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# patch: suppress daisy warnings
logging.getLogger('daisy.client').setLevel(logging.ERROR)


def create_mask_worker(
        block,
        raw,
        out,
        min_gray_value,
        max_gray_value):

    # load the chunk
    data = raw[block.read_roi].to_ndarray()

    # filter by value
    mask = ((data > min_gray_value) & (data < max_gray_value))

    # remove salt
    mask = skimage.morphology.binary_opening(mask, selem=ball(3))
    # remove pepper
    mask = skimage.morphology.binary_closing(mask, selem=ball(2))

    # TODO probably best to provide context for theses operations

    # remove larger holes
    mask = skimage.morphology.remove_small_holes(mask, area_threshold=100000)
    # remove larger objects
    mask = skimage.morphology.remove_small_objects(mask, min_size=100000)

    mask = mask.astype(np.uint8)

    # just for visualization
    mask *= 255

    # save to output dataset
    out[block.write_roi] = mask


def create_mask(
        filename,
        ds_name,
        out_ds_name,
        chunk_shape,
        min_gray_value,
        max_gray_value,
        num_workers):

    raw = open_ds(
        filename,
        ds_name,
        mode='r'
    )

    out = prepare_ds(
        filename=filename,
        ds_name=out_ds_name,
        total_roi=raw.roi,
        voxel_size=raw.voxel_size,
        dtype=raw.dtype,
        write_size=raw.voxel_size * Coordinate(chunk_shape),
        compressor={'id': 'zlib', 'level': 3}
    )

    # Spawn a worker per chunk
    block_roi = Roi(
        (0, 0, 0),
        raw.voxel_size * Coordinate(chunk_shape)
    )

    start = now()

    task = daisy.Task(
        total_roi=raw.roi,
        read_roi=block_roi,
        write_roi=block_roi,
        process_function=lambda block: create_mask_worker(
            block,
            raw,
            out,
            min_gray_value,
            max_gray_value
        ),
        read_write_conflict=False,
        fit='shrink',
        num_workers=num_workers,
        task_id="create_mask"
    )

    daisy.run_blockwise([task])

    logger.info(f"Done in {now() - start} s")


def parse_args():
    p = argparse.ArgParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add('--config', is_config_file=True, help='config file path')
    p.add(
        '--filename',
        '-f',
        required=True,
    )
    p.add(
        '--dataset',
        '-d',
        required=True,
        help='name of the raw dataset'
    )
    p.add(
        '--out_dataset',
        '-o',
        default='volumes/mask',
        help='name of the new mask dataset'
    )
    p.add(
        '--chunk_shape',
        '-c',
        nargs='+',
        type=int,
        default=[128, 128, 128],
        help='Size of a chunk in voxels. Should be a multiple of the existing chunk size. Bigger is better for hole filling, but slower'
    )
    p.add(
        '--min_gray_value',
        type=int,
        default=2,
        help='lower boundary for masking by value'
    )
    p.add(
        '--max_gray_value',
        type=int,
        default=180,
        help='upper boundary for masking by value'
    )
    p.add(
        '--num_workers',
        '-n',
        type=int,
        default=32,
        help='number of daisy processes'
    )

    args = p.parse_args()
    logger.info(f'\n{p.format_values()}')

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
        args.num_workers
    )


if __name__ == '__main__':
    main()
