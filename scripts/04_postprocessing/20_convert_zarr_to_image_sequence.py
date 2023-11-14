"""
Convert 3D zarr arrays to .tif image sequences
"""

import logging
import os
from time import time as now
import warnings

import configargparse as argparse
import numpy as np
import zarr
import skimage
from skimage import io

from funlib.persistence import Array, open_ds, prepare_ds
from funlib.geometry import Roi, Coordinate
import daisy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# patch: suppress daisy warnings
logging.getLogger('daisy.client').setLevel(logging.ERROR)


def convert_to_uint8(array):
    dtype = array.dtype

    # TODO: Remove hack to convert uint32 labels
    if np.dtype(dtype) == np.uint32:
        # logger.warning(
        # "Array contains uint32, all non-zero values will be set to 255.")
        array = (array != 0).astype(np.uint8) * 255
        return array

    # typecast ints
    elif np.issubdtype(dtype, np.integer):
        if array.max() > 255:
            raise ValueError(
                "Array contains integers >255, "
                "cannot safely convert to uint8."
            )
        return array.astype(np.uint8)

    # floats should be in [0,1] and are scaled to [0,255]
    elif np.issubdtype(dtype, np.float):
        if array.min() < 0.0 or array.max() > 1.0:
            raise ValueError(
                "Array contains floats outside [0,1], "
                "cannot safely scale to uint8."
            )
        array = skimage.img_as_ubyte(array)
        # array = array * 255.0

        return array.astype(np.uint8)

    else:
        raise TypeError(f"Conversion to uint8 not defined for dtype {dtype}.")


def convert_worker(block, ds, out_path):
    data = ds.to_ndarray(roi=block.read_roi)

    ds_offset_z = ds.roi.get_offset()[0] / ds.voxel_size[0]
    start_z = int(
        block.write_roi.get_offset()[0] / ds.voxel_size[0] -
        ds_offset_z
    )
    stop_z = int(
        (
            block.write_roi.get_offset()[0] + block.write_roi.get_shape()[0]
        ) / ds.voxel_size[0] - ds_offset_z
    )

    logger.debug(f"{ds_offset_z=}")
    logger.debug(f"{start_z=}")
    logger.debug(f"{stop_z=}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for section, z in zip(data, range(start_z, stop_z)):
            section = convert_to_uint8(section)
            io.imsave(
                os.path.join(out_path, f"section_{z:04d}.tif"),
                section,
                # compress=9
            )


def convert(filename, ds_name, out_path, num_workers):
    logger.info(
        f"Converting {os.path.join(filename, ds_name)}")
    start = now()

    shape = zarr.open(os.path.join(filename, ds_name), 'r').shape
    if not len(shape) == 3:
        raise NotImplementedError(
            "Conversion only implemented for 3D zarr arrays")

    ds = open_ds(
        filename,
        ds_name,
        mode='r'
    )

    if np.dtype(ds.dtype) != np.uint8:
        logger.warning(
            f'Input dtype {ds.dtype} does not match output dtype uint8.')

    chunk_shape = zarr.open(os.path.join(filename, ds_name), 'r').chunks

    # zyx format
    block_roi = Roi(
        (0, 0, 0),
        (ds.voxel_size[0] * chunk_shape[0],) + tuple(ds.roi.get_shape()[1:])

    )

    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    task = daisy.Task(
        total_roi=ds.roi,
        read_roi=block_roi,
        write_roi=block_roi,
        process_function=lambda block: convert_worker(
            block,
            ds,
            out_path,
        ),
        fit='shrink',
        read_write_conflict=False,
        num_workers=num_workers,
        task_id="convert_zarr_to_images"
    )

    daisy.run_blockwise([task])

    logger.info(
        f"Done in {now() - start} s")


def convert_zarr_to_image_sequences(
        filenames,
        datasets,
        out_directory,
        out_datasets,
        num_workers):

    assert len(filenames) == len(datasets), \
        'Provide a list of datasets for each filename.'
    assert len(filenames) == len(out_datasets), \
        "Provide a list of out_datasets for each filename."

    offset = None
    shape = None

    for f, list_of_ds, list_of_out_ds in zip(
            filenames, datasets, out_datasets):

        assert len(list_of_ds) == len(list_of_out_ds), \
            "Provide one out_dataset for each dataset name."

        for ds, out_ds in zip(
                list_of_ds, list_of_out_ds):
            zds = zarr.open(os.path.join(f, ds), 'r')

            # check for matching ROIs of all datasets
            try:
                new_offset = zds.attrs['offset']
                if offset is None:
                    offset = new_offset
                else:
                    if new_offset != offset:
                        logger.warning((
                            f"Offset {new_offset} of {os.path.join(f, ds)} "
                            f"does not match offset {offset} "
                            f"of previous datasets."
                        ))
            except KeyError:
                logger.warning(f"Cannot find offset for {os.path.join(f, ds)}")

            new_shape = zds.shape
            if shape is None:
                shape = new_shape
            else:
                if new_shape != shape:
                    logger.warning((
                        f"Shape {new_shape} of {os.path.join(f, ds)} "
                        f"does not match shape {shape} of previous datasets."
                    ))

            convert(
                filename=f,
                ds_name=ds,
                out_path=os.path.join(out_directory, out_ds),
                num_workers=num_workers
            )


def parse_args():
    p = argparse.ArgParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add('--config', is_config_file=True, help='config file path')
    p.add(
        '--filename',
        '-f',
        type=str_path,
        required=True,
        action='append',
        help=(
            'Path to the zarr file. '
            'You can convert dataset from multiple zarr files '
            'with multiple -f arguments.'
        )
    )
    p.add(
        '--datasets',
        '-d',
        type=str_rstrip_slash,
        required=True,
        nargs='+',
        action='append',
        help='The datasets in a zarr file to convert.'
    )
    p.add(
        '--out_directory',
        '-o',
        type=str_path,
        required=True,
        help=(
            "Name of the parent output directory that contains the image "
            " sequences."
        )
    )
    p.add(
        '--out_datasets',
        type=str_rstrip_slash,
        nargs='+',
        action='append',
        default=None,
        help=(
            'The datasets in a zarr file to convert. '
            'Defaults to the name of the input datasets.'
        )
    )
    p.add(
        '--num_workers',
        '-n',
        type=int,
        default=32
    )

    args = p.parse_args()
    logger.info(f'\n{p.format_values()}')

    if args.out_datasets is None:
        args.out_datasets = args.datasets

    return args


def str_rstrip_slash(x):
    return x.rstrip('/')

def str_path(x):
    return os.path.expanduser(x).rstrip('/')


def main():
    args = parse_args()

    convert_zarr_to_image_sequences(
        filenames=args.filename,
        datasets=args.datasets,
        out_directory=args.out_directory,
        out_datasets=args.out_datasets,
        num_workers=args.num_workers
    )


if __name__ == '__main__':
    main()