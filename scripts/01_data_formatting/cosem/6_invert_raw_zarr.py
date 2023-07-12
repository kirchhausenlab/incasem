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
        help='zarr path of volumes/raw, uint8 dataset, to invert')
    p.add(
        '-c',
        '--chunks',
        default=[256, 256, 256])
    args = p.parse_args()
    logger.info(f'\n{p.format_values()}')
    return args


def main():
    args = parse_args()

    fos.utils.invert_zarr(
        zarr_ds=args.zarr,
        chunks=args.chunks
        )

    logger.info(f'Inverted dataset created.')


if __name__ == '__main__':
    main()
