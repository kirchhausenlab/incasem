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
        default=['mito', 'golgi', 'er'],
        nargs='+')
    args = p.parse_args()
    logger.info(f'\n{p.format_values()}')
    return args


def main():
    args = parse_args()

    fos.utils.make_metric_masks_janelia_crops(
        zarr_ds=args.zarr,
        organelles=args.organelles,
        )

    logger.info(f'All metric masks have been created.')


if __name__ == '__main__':
    main()
