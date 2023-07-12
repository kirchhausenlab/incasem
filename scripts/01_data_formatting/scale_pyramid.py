import argparse

from incasem.utils import scale_pyramid

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create a scale pyramide for a zarr/N5 container.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--file',
        '-f',
        type=str,
        help="The input container")
    parser.add_argument(
        '--ds',
        '-d',
        type=str,
        help="The name of the dataset")
    parser.add_argument(
        '--scales',
        '-s',
        nargs='*',
        type=int,
        required=True,
        help="The downscaling factor between scales")
    parser.add_argument(
        '--chunk_shape',
        '-c',
        nargs='*',
        type=int,
        default=None,
        help="The size of a chunk in voxels")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    scale_pyramid(args.file, args.ds, args.scales, args.chunk_shape, args.num_workers)
