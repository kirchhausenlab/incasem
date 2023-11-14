'''Create a segmentation by thresholding a predicted probability map'''

import os
import logging
from time import time as now

import numpy as np
import configargparse as argparse

from funlib.persistence import Array, open_ds, prepare_ds
from funlib.geometry import Roi, Coordinate
import daisy


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# patch: suppress daisy warnings
logging.getLogger('daisy.client').setLevel(logging.ERROR)


def extract_segmentation_with_threshold_worker(
        block,
        probas,
        mask,
        out,
        threshold):

    # load the chunk
    probas = probas[block.read_roi].to_ndarray()
    if mask:
        mask = mask[block.read_roi].to_ndarray()
        segmentation = ((probas >= threshold) & (mask != 0))
    else:
        segmentation = probas >= threshold

    # store binary mask as {0,255}
    segmentation = segmentation.astype(np.uint32) * 255

    # save to output dataset
    out[block.write_roi] = segmentation


def extract_segmentation_with_threshold(
        filename,
        ds_name,
        mask_filename,
        mask_ds_name,
        out_ds_name,
        chunk_shape,
        threshold,
        num_workers):

    probas = open_ds(
        filename,
        ds_name,
        mode='r'
    )

    try:
        mask = open_ds(
            mask_filename,
            mask_ds_name,
            mode='r'
        )
    except (KeyError, RuntimeError):
        logger.warning((
            "Did not find a mask dataset "
            f"at {os.path.join(mask_filename, str(mask_ds_name))}."
        ))
        mask = None

    out = prepare_ds(
        filename=filename,
        ds_name=out_ds_name,
        total_roi=probas.roi,
        voxel_size=probas.voxel_size,
        dtype=np.uint32,
        write_size=probas.voxel_size * Coordinate(chunk_shape),
        compressor={'id': 'zlib', 'level': 3}
    )

    # Spawn a worker per chunk
    block_roi = Roi(
        (0, 0, 0),
        probas.voxel_size * Coordinate(chunk_shape)
    )

    start = now()

    task = daisy.Task(
        total_roi=probas.roi,
        read_roi=block_roi,
        write_roi=block_roi,
        process_function=lambda block: extract_segmentation_with_threshold_worker(
            block,
            probas=probas,
            mask=mask,
            out=out,
            threshold=threshold
        ),
        read_write_conflict=False,
        fit='shrink',
        num_workers=num_workers,
        task_id = "extract_segmentation_with_threshold"
    )

    daisy.run_blockwise([task])

    logger.info(f"Done in {now() - start} s")


def parse_args():
    p = argparse.ArgParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add('--config', is_config_file=True, help='config file path')
    p.add(
        '--prediction_filename',
        required=True,
        help="Zarr file with the prediction."
    )
    p.add(
        '--dataset',
        '-d',
        required=True,
        help='Name of the dataset with prediction probabilities.'
    )
    p.add(
        '--mask_filename',
        default="",
        help="Zarr file with the mask."
    )
    p.add(
        '--mask',
        '-m',
        default='volumes/mask',
        help='Binary mask to exclude non-cell voxels.'
    )
    p.add(
        '--out_dataset',
        '-o',
        required=True,
        help='Name of the output segmentation in the prediction zarr file.'
    )
    p.add(
        '--chunk_shape',
        '-c',
        nargs='+',
        type=int,
        default=[128, 128, 128],
        help=(
            'Size of a chunk in voxels. Should be a multiple of the existing '
            'chunk size.'
        )
    )
    p.add(
        '--threshold',
        '-t',
        type=float,
        required=True,
        help='Threshold for positive prediction.'
    )
    p.add(
        '--num_workers',
        '-n',
        type=int,
        default=32,
        help='Number of daisy processes.'
    )

    args = p.parse_args()
    logger.info(f'\n{p.format_values()}')

    return args


def main():
    args = parse_args()
    extract_segmentation_with_threshold(
        filename=args.prediction_filename,
        ds_name=args.dataset,
        mask_filename=args.mask_filename,
        mask_ds_name=args.mask,
        out_ds_name=args.out_dataset,
        chunk_shape=args.chunk_shape,
        threshold=args.threshold,
        num_workers=args.num_workers
    )


if __name__ == '__main__':
    main()