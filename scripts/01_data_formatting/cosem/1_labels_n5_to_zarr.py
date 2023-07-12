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
        '-n',
        '--n5',
        required=True,
        type=str,
        help='n5 dataset to convert')
    p.add(
        '-z',
        '--zarr',
        required=True,
        type=str,
        help='zarr file to be created')
    p.add(
        '--chunk_size',
        default=256,
        type=int)
    p.add(
        '--offset',
        default=[0, 0, 0])
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

    logger.info((
        f'\nConverting from:\n\t{args.n5}'
        f'\nto:\n\t{args.zarr}'
    ))

    fos.utils.n5_to_zarr_gt(
        args.n5,
        args.zarr,
        chunk_size=args.chunk_size,
        offset=args.offset,
        resolution=args.resolution,
        workers=args.workers,
        threads_per_worker=args.threads_per_worker
        )

    logger.info(f'All .zarr arrays have been created.')


if __name__ == '__main__':
    main()
