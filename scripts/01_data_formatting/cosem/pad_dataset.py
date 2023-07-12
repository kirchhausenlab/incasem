import logging
import configargparse as argparse

import incasem as fos

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def parse_args():
    p = argparse.ArgParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add(
        '-z',
        '--zarr',
        required=True,
        type=str,
        help='zarr filename, eg. /nfs/../hela_2.zarr')
    p.add(
        '-d',
        '--ds',
        required=True,
        type=str,
        help='dataset to pad, eg. volumes/raw')
    p.add(
        '-p',
        '--pad',
        type=int,
        required=True,
        nargs='+',
        help='Padding voxels to add along zyx, before and after the array along each axis, eg. 50 50 20 30 0 0, will add along y 20 voxels before the array and 30 after the array')
    p.add(
        '-c',
        '--chunks',
        default=[256, 256, 256])
    p.add(
        '--num_workers',
        type=int,
        default=32)
    p.add(
        '--threads_per_worker',
        type=int,
        default=3)
    args = p.parse_args()
    logger.info(f'\n{p.format_values()}')
    return args


def main():
    args = parse_args()

    fos.utils.pad_dataset(
        zarr=args.zarr,
        ds=args.ds,
        pad=args.pad,
        chunks=args.chunks,
        workers=args.num_workers,
        threads_per_worker=args.threads_per_worker
        )

    logger.info(f'Padded dataset created.')


if __name__ == '__main__':
    main()
