"""
Call downsample script on all labels and metric mask
of a zarr file, for a given organelles list.

Example:
python downsample_label_metric_mask.py -f /nfs/scratch2/fiborganelles/data/janelia/jrc_hela-2/jrc_hela-2.zarr -o /nfs/scratch2/fiborganelles/data/8nm/janelia/jrc_hela-2/jrc_hela-2.zarr --factors 2 2 2  --interpolatable False
"""
import logging

import configargparse as argparse
import os

import incasem as fos


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def str_rstrip_slash(x):
    return x.rstrip('/')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_crops(filename, organelles, what='labels'):
    crop_list = []
    if what == 'metric_masks':
        organelles.remove('background')

    logger.debug(f'{what=}')
    for organelle in organelles:
        logger.debug(f'{organelle}')
        base_dir = os.path.join(filename, 'volumes', what, organelle)
        os.chdir(base_dir)
        for file_or_folder in os.listdir():
            full_path = os.path.join(base_dir, file_or_folder)
            if not os.path.isdir(full_path):
                continue
            if file_or_folder[0] == '.':
                continue
            crop = file_or_folder
            logger.debug(f'   {crop}')
            crop_list.append(os.path.join('volumes',
                                          what,
                                          organelle,
                                          crop))
    return crop_list


def parse_args():
    p = argparse.ArgParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add(
        '--filename',
        '-f',
        type=str_rstrip_slash,
        required=True,
    )
    p.add(
        '--out_filename',
        '-o',
        type=str_rstrip_slash,
        required=True
    )
    p.add(
        '--organelles',
        default=['er', 'mito', 'golgi', 'background'],
        nargs='+',
        help='Organelles whose labels and metric masks crops are downsampled'
    )
    p.add(
        '--factors',
        type=int,
        nargs='+',
        required=True,
        help='Downscaling factors for each dimension, zyx.'
    )
    p.add(
        '--interpolatable',
        nargs='+',
        type=str2bool,
        required=True,
        help='Indicate for each dataset whether values can be interpolated.'
    )
    p.add(
        '--method',
        choices=['simulate_em_low_res', 'downscale_local_mean'],
        default='simulate_em_low_res',
        help=(
            '`simulate_em_low_res`: Mean downscaling xy, slicing in z. --- '
            '`downscale_local_mean`: Mean downscaling in all dimensions.'
        )
    )
    p.add(
        '--chunk_shape',
        '-c',
        nargs='+',
        type=int,
        default=[128, 128, 128],
        help='Size of a output chunks in voxels'
    )
    p.add(
        '--num_workers',
        '-n',
        type=int,
        default=32,
        help="Number of parallel processes."
    )

    args = p.parse_args()
    logger.info(f'\n{p.format_values()}')

    return args


def join_list(a_list):
    return " ".join(str(x) for x in a_list)


def main():
    args = parse_args()
    label_list = get_crops(args.filename, args.organelles, what='labels')
    logger.debug(f'{label_list=}')
    metric_mask_list = get_crops(args.filename, args.organelles, what='metric_masks')
    logger.debug(f'{metric_mask_list=}')

    all_crops = label_list + metric_mask_list

    logger.debug(f'to downscale: {all_crops}')

    for crop in all_crops:

        cmd = f"python ~/code/incasem/scripts/01_data_formatting/downscale.py -f {args.filename} -o {args.out_filename} -d {crop} --factors {join_list(args.factors)} --interpolatable {join_list(args.interpolatable)}"
        
        logger.debug(cmd)

        os.system(cmd)

    # if run directly, most blocks
    # will be skipped by Daisy. For now we simply
    # call the python script from the
    # shell iteratively, even if it's
    # quite dumb

    # fos.utils.downscale(
    #     args.filename,
    #     args.out_filename,
    #     all_crops,
    #     all_crops,
    #     args.factors,
    #     args.interpolatable,
    #     args.method,
    #     args.chunk_shape,
    #     args.num_workers)


if __name__ == '__main__':
    main()
