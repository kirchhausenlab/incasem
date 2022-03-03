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
        '-i',
        '--in_arrays',
        required=True,
        nargs='+',
        help='List of n5 arrays to convert, e.g.  \
        "em/s0, labels/mito_pm_contacts/s0"')
    p.add(
        '-o',
        '--out_arrays',
        required=True,
        nargs='+',
        help='List of .zarr arrays to create \
        from the corresponding n5 inputs, e.g. \
        "volumes/raw, volumes/labels/mito"')
    p.add(
        '--chunk_size',
        default=256,
        type=int)
    p.add(
        '--offset',
        default=[0, 0, 0])
    p.add(
        '--resolution',
        default=[5.24, 4, 4])
    p.add(
        '--workers',
        default=32,
        type=int)
    p.add(
        '--threads_per_worker',
        default=3,
        type=int)
    p.add(
        '--raw_same_bit',
        action='store_true',
        help='Conserve raw data bit depth \
        instead of converting it to 8 bit. By default, False.')
    args = p.parse_args()
    logger.info(f'\n{p.format_values()}')
    return args


def main():
    args = parse_args()

    logger.info((
        f'\nConverting from:\n\t{args.n5}'
        f'\nto:\n\t{args.zarr}'
    ))

    assert len(args.in_arrays) == len(args.out_arrays), \
        'input n5 arrays and output zarr arrays are not the same number'

    n5_and_zarr_arrays = list(zip(list(args.in_arrays), list(args.out_arrays)))

    fos.utils.n5_to_zarr(
        args.n5,
        args.zarr,
        n5_and_zarr_arrays,
        chunk_size=args.chunk_size,
        offset=args.offset,
        resolution=args.resolution,
        workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        raw_8_bit=not(args.raw_same_bit))

    logger.info(f'All .zarr arrays have been created.')


if __name__ == '__main__':
    main()
