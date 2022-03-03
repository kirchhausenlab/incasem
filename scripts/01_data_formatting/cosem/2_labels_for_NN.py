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
        help='zarr dataset whose groundtruth will b e processed')
    p.add(
        '-o',
        '--organelles',
        default=['mito', 'golgi', 'er', 'np'],
        nargs='+')
    p.add(
        '--resolution',
        default=[4, 4, 4])
    p.add(
        '--workers',
        default=32,
        type=int)
    p.add(
        '--threads_per_worker',
        default=3,
        type=int)
    args = p.parse_args()
    logger.info(f'\n{p.format_values()}')
    return args


def main():
    args = parse_args()

    fos.utils.gt_to_organelle(
        zarr_ds=args.zarr,
        organelles=args.organelles,
        resolution=args.resolution,
        workers=args.workers,
        threads_per_worker=args.threads_per_worker
        )

    logger.info(f'All .zarr arrays have been created.')


if __name__ == '__main__':
    main()
