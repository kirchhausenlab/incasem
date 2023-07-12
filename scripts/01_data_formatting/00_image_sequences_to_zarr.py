"""Convert multiple sequences of images
that belong together to block-wise storage
"""

import logging
import configargparse as argparse

import incasem as fos

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# patch: suppress daisy warnings
logging.getLogger('daisy.client').setLevel(logging.ERROR)


def parse_args():
    ''' TODO '''
    p = argparse.ArgParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add(
        '-c',
        '--config',
        is_config_file=True,
        help='config file path')
    p.add(
        '-i',
        '--input_dir',
        nargs='+',
        required=True,
        help='Directions with section images')
    p.add(
        '-f',
        '--output_file',
        required=True,
        help='Zarr file to be created')
    p.add(
        '-d',
        '--output_dataset',
        nargs='+',
        default=['volumes/raw'],
        help='datasets inside zarr file')
    p.add(
        '-r',
        '--image_regex',
        default=r'.*_(\d+).*\.tif$',
        help='regex to select sections and extract their numbers')
    p.add(
        '--dtype',
        nargs='+',
        default=['uint8'],
        help='any numpy datatype')
    p.add(
        '--offset',
        default=[0, 0, 0],
        type=int,
        nargs=3,
        help='z,y,x offset in voxels')
    p.add(
        '--shape',
        default=[None, None, None],
        type=int,
        nargs=3,
        help='z,y,x shape in voxels')
    p.add(
        '--resolution',
        default=[5, 5, 5],
        type=int,
        nargs=3,
        help='z,y,x resolution in nanometers')
    p.add(
        '--chunks',
        default=[128, 128, 128],
        type=int,
        nargs=3,
        help='z,y,x chunk size in voxels')
    p.add(
        '--conversion',
        type=str_or_none,
        nargs='+',
        default=[None],
        help='conversion of each slice, using Pillow')
    p.add(
        '--invert',
        action='store_true',
        help='Invert raw data LUT')
    p.add(
        '--num_workers',
        default=32,
        type=int)

    args = p.parse_args()
    logger.info(f'\n{p.format_values()}')

    return args


def str_or_none(x):
    """Type to pass python `None` via argparse

    Args:
        x (str)

    Returns:
        str or `None`
    """
    return None if x == 'None' else x


def main():
    args = parse_args()

    for i in range(len(args.input_dir)):
        logger.info((
            f'Converting from {args.input_dir[i]} '
            f'to {args.output_dataset[i]}...'
        ))
        fos.utils.image_sequence_to_zarr(
            args.input_dir[i],
            args.output_file,
            args.output_dataset[i],
            args.image_regex,
            args.dtype[i],
            args.offset,
            args.shape,
            args.resolution,
            args.chunks,
            args.conversion[i],
            args.num_workers,
            invert=args.invert)


if __name__ == '__main__':
    main()
