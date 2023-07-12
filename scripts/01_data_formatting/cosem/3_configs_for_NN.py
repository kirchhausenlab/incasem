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
    p.add(
        '-o',
        '--organelles',
        default=['mito', 'golgi', 'er', 'background'],
        nargs='+')
    args = p.parse_args()
    logger.info(f'\n{p.format_values()}')
    return args


def main():
    args = parse_args()

    fos.utils.make_config_janelia_crops(
        zarr_ds=args.zarr,
        organelles=args.organelles,
        )

    logger.info(f'All configs have been created.')


if __name__ == '__main__':
    main()
