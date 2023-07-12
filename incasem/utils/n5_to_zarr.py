import logging
import os
import json

from fibsem_tools.io import read

import dask
import dask.array as da
from dask.distributed import Client

import zarr
from numcodecs import Zlib

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def n5_to_zarr_raw(n5_path: str,
                   zarr_path: str,
                   arrays: list,
                   chunk_size=256,
                   offset=[0, 0, 0],
                   resolution=[4, 4, 4],
                   workers=32,
                   threads_per_worker=3,
                   raw_8_bit=True):

    client = init_client(workers, threads_per_worker)
    dask_address = client.scheduler_info()["services"]["dashboard"]
    logger.info(f'\n\nDask dashboard at: http://localhost:{dask_address}\n\n')

    for arr_in, arr_out in arrays:
        dataset_n5_to_zarr(
            n5_path,
            zarr_path,
            arr_in,
            arr_out,
            chunk_size=chunk_size,
            offset=offset,
            resolution=resolution,
            raw_8_bit=raw_8_bit)


def n5_to_zarr_gt(n5_path: str,
                  zarr_path: str,
                  chunk_size=256,
                  offset=[0, 0, 0],
                  resolution=[4, 4, 4],
                  workers=32,
                  threads_per_worker=3):

    client = init_client(workers, threads_per_worker)
    logger.info(f'\n\nDask dashboard at:\n\n \
            http://localhost:{client.scheduler_info()["services"]["dashboard"]}\n\n')

    arrays = crops_paths(n5_path)

    for arr_in, arr_out in arrays:
        
        raw_shape = shape_raw_from_meta(
            os.path.join(zarr_path,
                         'volumes',
                         'raw',
                         '.zarray')
            )

        dataset_n5_to_zarr(
            n5_path,
            zarr_path,
            arr_in,
            arr_out,
            chunk_size=chunk_size,
            offset=offset,
            resolution=resolution)

        metadata_n5_to_zarr(
            os.path.join(n5_path, arr_in),
            os.path.join(zarr_path, arr_out),
            resolution,
            raw_shape)

def shape_raw_from_meta(zarr_path):
    raw_shape = []  #  x, y, z
    with open(zarr_path, 'r') as json_meta:
        raw_meta = json.load(json_meta)
        raw_shape = raw_meta['shape'][::-1]
    return raw_shape


def metadata_n5_to_zarr(
        n5_arr: str,
        zarr_arr: str,
        resolution: list,
        raw_shape: list):

    res_x = resolution[0]
    res_y = resolution[1]
    res_z = resolution[2]

    n5_metadata = os.path.join(n5_arr, 'attributes.json')

    zarr_attrs = {}
    with open(n5_metadata, 'r') as infile:
        n5_meta = json.load(infile)

        y_offset = 0
        if 'jurkat' in n5_arr:
            y_offset = 10
        zarr_attrs['offset'] = [int(n5_meta['offset'][2]*0.25),
                                raw_shape[1] - int(n5_meta['offset'][1]*0.25)
                                - int(n5_meta['dimensions'][1]*0.5)
                                - y_offset,
                                int(n5_meta['offset'][0]*0.25)]

        zarr_attrs['offset'][0] *= res_z
        zarr_attrs['offset'][1] *= res_y
        zarr_attrs['offset'][2] *= res_x

        zarr_attrs['resolution'] = resolution

    zarr_attrs_file = os.path.join(zarr_arr, '.zattrs')
    with open(zarr_attrs_file, 'w') as outfile:
        logger.info(f'  saving to {outfile}')
        logger.info(zarr_attrs)
        json.dump(zarr_attrs, outfile)

def dataset_n5_to_zarr(n5_path: str,
                       zarr_path: str,
                       arr_n5: str,
                       arr_zarr: str,
                       chunk_size=256,
                       offset=[0, 0, 0],
                       resolution=[1, 1, 1],
                       raw_8_bit=True):

    n5_arr_path = os.path.join(n5_path, arr_n5)
    logger.info(f'  reading {n5_arr_path}')
    n5_arr = read(n5_arr_path)
    data = da.from_array(n5_arr,
                         chunks=(chunk_size, chunk_size, chunk_size),
                         asarray=True)

    assert any(el in arr_zarr for el in ['raw', 'groundtruth'])

    if 'groundtruth' in arr_n5:
        data_int8 = data[::2, ::2, ::2].astype('u1')
        da.rechunk(data_int8)
        data_int8 = da.flip(data_int8, 1)
        data_int8 = da.rechunk(data_int8)
    if 'raw' in arr_zarr:
        if raw_8_bit:
            data_int8 = (data/256).astype('u1')
        else:
            data_int8 = data

    zarr_ds_path = create_zarr(zarr_path,
                               arr_zarr,
                               shape=data_int8.shape,
                               chunk_size=chunk_size,
                               offset=offset,
                               resolution=resolution)

    logger.info(f'\n\n\t creating: {arr_zarr}\n')
    da.to_zarr(data_int8,
               zarr_ds_path,
               component=None,
               overwrite=True,
               compute=True,
               return_stored=False,
               compressor=Zlib(level=3))


def crops_paths(n5_path,
                n5_prefix='volumes/groundtruth/0003',
                n5_suffix='labels/all',
                zarr_suffix='volumes/groundtruth'):

    crops_base_dir = os.path.join(n5_path, n5_prefix)
    logger.debug(crops_base_dir)

    crops_names = [c for c in os.listdir(crops_base_dir) if 'crop' in c]  # 'Crop3', 'Crop25'..
    logger.info(f'crops to convert \n: {crops_names}')

    crops_arrs = []
    dest_arrs = []

    for crop_name in crops_names:
        crops_arrs.append(
            os.path.join(
                n5_prefix,
                crop_name,
                n5_suffix)
        )

        dest_arrs.append(
            os.path.join(
                zarr_suffix,
                crop_name.replace('crop', '')
            )
        )

    return list(zip(crops_arrs, dest_arrs))


def create_zarr(zarr_path,
                arr_zarr,
                shape,
                chunk_size=256,
                offset=[0, 0, 0],
                resolution=[5.24, 4, 4]):
    """
    Create an empty zarr container containing an empty array
    of specified shape and chunk size.
    """
    root = zarr.group(store=zarr_path, overwrite=False)
    groups_path, dataset_name = os.path.split(arr_zarr)
    groups = groups_path.split('/')

    current_path = zarr_path
    for level, next_group in enumerate(groups):
        current_path = os.path.join(current_path, next_group)
        try:
            group = zarr.open_group(store=current_path, mode='r')
        except zarr.errors.GroupNotFoundError:
            group = zarr.open_group(store=current_path, mode='w')
    group._read_only = False
    dataset = group.create_dataset(dataset_name,
                                   overwrite=True,
                                   compressor=Zlib(level=3),
                                   shape=shape,
                                   chunks=(chunk_size, chunk_size, chunk_size),
                                   dtype='u1')
    dataset.attrs['offset'] = offset
    dataset.attrs['resolution'] = resolution

    dataset_path = os.path.join(zarr_path, arr_zarr)
    return dataset_path


def init_client(workers, threads_per_worker):
    client = Client(threads_per_worker=threads_per_worker, n_workers=workers)
    return client


def mask_from_gt(gt, no_label_index):
    '''
    Return mask from Janlia ground truth
    dataset. Voxels of 0s are not labelled,
    so we create a mask not to calculate the loss there.
    '''
    gt_5nm = gt[::2, ::2, ::2]
    return da.where(gt_5nm == no_label_index, 0, 255).astype('u1')
