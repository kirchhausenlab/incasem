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
        help='zarr data and label for configs')
    args = p.parse_args()
    logger.info(f'\n{p.format_values()}')
    return args


def main():
    args = parse_args()

    fos.utils.make_mask_janelia_crops(
        zarr_ds=args.zarr,
        )

    logger.info(f'Mask created.')


if __name__ == '__main__':
    main()
