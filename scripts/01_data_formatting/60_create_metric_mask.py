'''Create a mask to ignore predictions at the boundary of objects'''

import logging
from time import time as now

import numpy as np
import configargparse as argparse
from scipy import ndimage

from funlib.persistence import Array, open_ds, prepare_ds
from funlib.geometry import Roi, Coordinate
import daisy


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# patch: suppress daisy warnings
logging.getLogger('daisy.client').setLevel(logging.ERROR)


def create_metric_mask_worker(
        block,
        labels,
        out,
        exclude_voxels_outwards,
        exclude_voxels_inwards):

    # load the chunk
    data = labels.to_ndarray(roi=block.read_roi, fill_value=0)

    # binarize
    data = (data != 0).astype(np.uint8)

    # dilate
    dilated = ndimage.binary_dilation(
        data, iterations=exclude_voxels_outwards).astype(
        np.uint8)

    # erode
    eroded = ndimage.binary_erosion(
        data, iterations=exclude_voxels_inwards).astype(
        np.uint8)

    # get boundary mask
    boundary_mask = dilated - eroded
    # invert
    mask = np.logical_not(boundary_mask).astype(np.uint8)
    # set to 0, 255
    mask *= 255

    # save to output dataset
    mask = Array(mask, roi=block.read_roi, voxel_size=labels.voxel_size)
    out[block.write_roi] = mask[block.write_roi]


def create_metric_mask(
        filename,
        ds_name,
        out_ds_name,
        chunk_shape,
        exclude_voxels_outwards,
        exclude_voxels_inwards,
        num_workers):

    labels = open_ds(
        filename,
        ds_name,
        mode='r'
    )

    out = prepare_ds(
        filename=filename,
        ds_name=out_ds_name,
        total_roi=labels.roi,
        voxel_size=labels.voxel_size,
        dtype=np.uint8,
        write_size=labels.voxel_size * Coordinate(chunk_shape),
        compressor={'id': 'zlib', 'level': 3}
    )

    write_roi = Roi(
        (0, 0, 0),
        labels.voxel_size * Coordinate(chunk_shape)
    )

    context = labels.voxel_size * \
        max(exclude_voxels_outwards, exclude_voxels_inwards)
    read_roi = write_roi.grow(context, context)

    start = now()

    task = daisy.Task(
        total_roi=labels.roi.grow(context, context),
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=lambda block: create_metric_mask_worker(
            block,
            labels,
            out,
            exclude_voxels_outwards,
            exclude_voxels_inwards,
        ),
        read_write_conflict=False,
        fit='shrink',
        num_workers=num_workers,
        task_id='create_metric_mask'
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
        help='Labels dataset to use for masking'
    )
    p.add(
        '--out_dataset',
        '-o',
        required=True,
        help='Name of the new metric mask dataset'
    )
    p.add(
        '--chunk_shape',
        '-c',
        nargs='+',
        type=int,
        default=[128, 128, 128],
        help='Size of an output chunk in voxels'
    )
    p.add(
        '--exclude_voxels_outwards',
        type=int,
        default=4,
        help="Dilate the labels by this many voxels"
    )
    p.add(
        '--exclude_voxels_inwards',
        type=int,
        default=4,
        help="Erode the labels by this many voxels"
    )
    p.add(
        '--num_workers',
        '-n',
        type=int,
        default=32,
        help='Number of daisy processes'

    )

    args = p.parse_args()
    logger.info(f'\n{p.format_values()}')

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
        args.num_workers
    )


if __name__ == '__main__':
    main()