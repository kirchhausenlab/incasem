"""Apply contrast limited adaptive histogram equalization"""

import logging
from time import time as now

import numpy as np
import configargparse as argparse

from funlib.persistence import Array, open_ds, prepare_ds
from funlib.geometry import Roi, Coordinate
import daisy

from incasem.utils import equalize_adapthist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# patch: suppress daisy warnings
logging.getLogger('daisy.client').setLevel(logging.ERROR)


def clahe_worker(
        block,
        raw,
        out,
        clahe_kernel_size,
        clahe_clip_limit
):
    # load the write roi and get its mean
    write_data = raw.to_ndarray(roi=block.write_roi)
    mean = int(np.mean(write_data))

    # load the chunk
    data = raw.to_ndarray(roi=block.read_roi, fill_value=mean)

    # apply clahe
    equalized = equalize_adapthist(
        image=data,
        kernel_size=clahe_kernel_size,
        clip_limit=clahe_clip_limit,
        nbins=256
    )
    assert equalized.dtype == np.uint8

    # save to output dataset
    equalized = Array(
        equalized,
        roi=block.read_roi,
        voxel_size=raw.voxel_size
    )
    out[block.write_roi] = equalized[block.write_roi]


def equalize_histogram(
        filename,
        ds_name,
        out_ds_name,
        chunk_shape,
        kernel_size,
        clip_limit,
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
    write_roi = Roi(
        (0, 0, 0),
        raw.voxel_size * Coordinate(chunk_shape)
    )

    # Add (1,1,1) to avoid division of odd number
    #context = ((raw.voxel_size * kernel_size) + (1,) * raw.voxel_size.dims) / 2
    context = (raw.voxel_size * kernel_size + Coordinate((1,) * raw.voxel_size.dims)) / 2
    read_roi = write_roi.grow(context, context)

    total_roi = raw.roi.grow(context, context)
    start = now()

    task = daisy.Task(
        total_roi=total_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=lambda block: clahe_worker(
            block,
            raw,
            out,
            kernel_size,
            clip_limit
        ),
        read_write_conflict=False,
        fit='shrink',
        num_workers=num_workers,
        task_id='histogram_equalization'
    )

    daisy.run_blockwise([task])

    logger.info(f'Done in {now() - start} s')


def parse_args():
    p = argparse.ArgParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add('--config', is_config_file=True, help='Config file path.')
    p.add(
        '--filename',
        '-f',
        required=True,
    )
    p.add(
        '--dataset',
        '-d',
        default='volumes/raw',
        help='Name of the raw dataset.'
    )
    p.add(
        '--out_dataset',
        '-o',
        default='volumes/raw_equalized_0.02',
        help='Name of the new output dataset.'
    )
    p.add(
        '--chunk_shape',
        '-c',
        nargs='+',
        type=int,
        default=[128, 128, 128],
        help='Size of a chunk in voxels.'
    )
    p.add(
        '--kernel_size',
        type=int,
        default=128,
        help='Block edge length for CLAHE.'
    )
    p.add(
        '--clip_limit',
        type=float,
        default=0.02,
        help='Clip relative frequency for adapting the histogram.'
    )
    p.add(
        '--num_workers',
        '-n',
        type=int,
        default=20,
        help='Number of daisy processes.'
    )

    args = p.parse_args()
    logger.info(f'\n{p.format_values()}')

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