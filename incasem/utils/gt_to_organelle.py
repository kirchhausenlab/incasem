import logging
import os
import shutil
import json

import dask.array as da
from dask.distributed import Client
import zarr
import numpy as np

from numcodecs import Zlib
from fibsem_tools.io import read

import incasem as fos


logger = logging.getLogger(__name__)


def gt_to_organelle(zarr_ds: str,
                    organelles: list,
                    workers=32,
                    threads_per_worker=3,
                    resolution=[4, 4, 4]):

    client = Client(processes=False)
#      client = init_client(workers, threads_per_worker)
#      dask_address = client.scheduler_info()["services"]["dashboard"]
#      logger.info(f'\n\nDask dashboard at: http://localhost:{dask_address}\n\n')
#
    gt_base_folder = os.path.join(
        zarr_ds,
        'volumes',
        'groundtruth')
    os.chdir(gt_base_folder)

    for ds in os.listdir():
        gt_ds = os.path.join(gt_base_folder, ds)
        if not os.path.isdir(gt_ds):
            continue

        with open(os.path.join(gt_ds, '.zattrs')) as f:
            meta = json.load(f)  # offset and resolution

        logger.info(f'Processing: {gt_ds}')
        create_label(
            gt_ds,
            organelles,
            zarr_ds,
            ds,  #  crop number
            meta,
            resolution=resolution)


def create_label(gt_ds: str,
                 organelles: str,
                 zarr_ds: str,
                 ds: str,
                 meta: dict,
                 resolution=[4, 4, 4],
                 chunks=(256, 256, 256)):
    gt_array = read(gt_ds)
    data = da.from_array(gt_array,
                         chunks=chunks,
                         asarray=True)

    labels = gt_to_labels(data, organelles)

    for organelle in labels:  # mito, golgi, er, background
        zarr_arr_path = os.path.join(
            'volumes',
            'labels',
            organelle,
            ds)

        data_to_save = labels[organelle].copy().astype('uint8')
        #  note: if you use chunk different than shape,
        #  neuroglancer won't read these
        #  array correctly
        chunks = data_to_save.shape

        zarr_data = da.from_array(data_to_save,
                                  chunks=chunks, asarray=True)

        zarr_ds_path = fos.utils.create_zarr(zarr_ds,
                                             zarr_arr_path,
                                             shape=zarr_data.shape,
                                             chunk_size=chunks[0],
                                             offset=meta['offset'],
                                             resolution=resolution)

        logger.info(f'\n\n\t creating: {zarr_ds_path}\n')

        da.to_zarr(
            zarr_data,
            zarr_ds_path,
            component=None,
            storage_options=None,
            compute=True,
            overwrite=True,
            return_stored=False,
            compressor=Zlib(level=3))

        #  zarr_arrs = os.path.normpath(zarr_arr_path).split(os.path.sep)
        #  zarr_base = zarr.open(zarr_ds,
        #                        mode='a',
        #                        shape=zarr_data.shape,
        #                        chunks=chunks,
        #                        dtype='|u1')
        #  print(zarr_base)
        #  print(zarr_arrs)
        #  zarr_group = zarr_base
        #  for arr in zarr_arrs:
        #  zarr_group = zarr_group[arr]
        #  zarr_group = zarr_data.astype('uint8')

        shutil.copy(
            os.path.join(gt_ds, '.zattrs'),
            os.path.join(zarr_ds, zarr_arr_path, '.zattrs')
        )
        shutil.copy(
            os.path.join(gt_ds, '.zarray'),
            os.path.join(zarr_ds, zarr_arr_path, '.zarray')
        )


def gt_to_labels(gt, organelles):
    '''
    From Janelia grountruth, create as many datasets
    as detected classes (mito, golgi, er, np, background)
    '''

    gt_np = gt.compute()

    gt_to_label_map = {
        'mito': [3, 4],
        'golgi': [6, 7],
        'er': [16, 17],
        'np': [22, 23]}
    label_ds = {}

    for org in organelles:
        print(org)
        org_in_gt = np.isin(gt_np, gt_to_label_map[org]).any()
        if org_in_gt:
            label_ds[org] = np.where(np.logical_or(
                gt_np == gt_to_label_map[org][0],
                gt_np == gt_to_label_map[org][1]),
                                     255, 0).astype('u1')

    if not label_ds:  # organelle not found, block is background
        label_ds['background'] = np.zeros_like(gt_np)

    logger.info(f'label found: {label_ds.keys()}')

    return label_ds


def init_client(workers, threads_per_worker):
    client = Client(threads_per_worker=threads_per_worker, n_workers=workers)
    return client
