import logging
import os
import shutil
import json

from fibsem_tools.io import read

import dask.array as da
import numpy as np

from numcodecs import Zlib

import incasem as fos


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def make_config_janelia_crops(zarr_ds: str,
                              organelles: list):

    zarr_ds = zarr_ds.rstrip('\\')  # remove trailing backslash if present
    zarr_path_split = os.path.normpath(zarr_ds).split(os.path.sep)

    zarr_path_to_data = os.path.join(*zarr_path_split[-3:])

    for org in organelles:
        crop_folder = os.path.join(
            zarr_ds,
            'volumes',
            'labels',
            org)
        os.chdir(crop_folder)

        config = {}
        for crop in os.listdir():

            section = org + '_' + crop

            crop_path = os.path.join(crop_folder, crop)
            if not os.path.isdir(crop_path):
                continue
            if crop[0] == '.':
                continue

            config[section] = {}

            with open(os.path.join(crop_path, '.zattrs')) as f:
                zattrs = json.load(f)  # offset and resolution
            with open(os.path.join(crop_path, '.zarray')) as f:
                zarray = json.load(f)  # shape and voxel size

            offset = zattrs['offset']
            res = zattrs['resolution']

            config[section]['file'] = zarr_path_to_data
            config[section]['offset'] = [int(offset[0]/res[0]),
                                         int(offset[1]/res[1]),
                                         int(offset[2]/res[2])]
            config[section]['shape'] = zarray['shape']
            config[section]['voxel_size'] = zattrs['resolution']
            config[section]['raw'] = 'volumes/raw_equalized_0.02'
            config[section]['labels'] = {'volumes/labels/'+org+'/'+crop: 1}
            config[section]['mask'] = 'volumes/mask'

            if org in ['mito', 'er', 'golgi']:
                config[section]['metric_masks'] = ['volumes/metric_masks/'
                                                   + org + '/' + crop]

        config_path = os.path.join(
                        os.path.dirname(zarr_ds),
                        org+'_config.json')
        with open(config_path, 'w') as outfile:
            json.dump(config, outfile, indent=4, sort_keys=True)

        logger.info(f'created: {config_path}')


def make_metric_masks_janelia_crops(zarr_ds: str,
                                    organelles: list):

    zarr_ds = zarr_ds.rstrip('\\')  # remove trailing backslash if present

    for org in organelles:
        crop_folder = os.path.join(
            zarr_ds,
            'volumes',
            'labels',
            org)
        os.chdir(crop_folder)

        crop_ds = []
        for crop in os.listdir():
            crop_path = os.path.join(crop_folder, crop)
            if not os.path.isdir(crop_path):
                continue

            crop_ds = os.path.join(
                'volumes',
                'labels',
                org,
                crop)

            metric_mask_ds = os.path.join(
                'volumes',
                'metric_masks',
                org,
                crop)

            #  make metric mask for this cro
            if org == 'mito':
                voxel_to_exclude = 4
            else:
                voxel_to_exclude = 2

            os.system(f"python ~/code/incasem/scripts/01_data_formatting/60_create_metric_mask.py -f {zarr_ds} -d {crop_ds} --out_dataset {metric_mask_ds} --exclude_voxels_inwards {voxel_to_exclude} --exclude_voxels_outwards {voxel_to_exclude}")
            logger.info(f'created: {metric_mask_ds}')


def get_crops_coords(zarr_ds: str):
    zarr_ds = zarr_ds.rstrip('\\')

    labels_offset = []
    labels_shape = []

    label_path = os.path.join(
                zarr_ds,
                'volumes',
                'labels')
    os.chdir(label_path)
    for org in os.listdir():
        org_path = os.path.join(label_path, org)
        if not os.path.isdir(org_path) or org[0] == '.':
            continue

        os.chdir(org_path)
        for crop in os.listdir():
            crop_path = os.path.join(org_path, crop)
            if not os.path.isdir(crop_path) or crop[0] == '.':
                continue


            with open(os.path.join(crop_path, '.zattrs')) as f:
                zattrs = json.load(f)  # offset and resolution
            with open(os.path.join(crop_path, '.zarray')) as f:
                zarray = json.load(f)  # shape and voxel size

            offset_nm = zattrs['offset']
            resolution = zattrs['resolution']

            offset = [int(offset_nm[0] / resolution[0]),
                      int(offset_nm[1] / resolution[1]),
                      int(offset_nm[2] / resolution[2])]  # z, y, x

            shape = zarray['shape']  # z, y, x

            labels_offset.append(offset)
            labels_shape.append(shape)

    return labels_offset, labels_shape


def make_empty_like_raw(zarr_ds: str, array_to_make: str):
    
    raw_path = os.path.join(zarr_ds, 'volumes', 'raw')
    with open(os.path.join(raw_path, '.zattrs')) as f:
        zattrs = json.load(f)
    with open(os.path.join(raw_path, '.zarray')) as f:
        zarray = json.load(f)

    #  create empty mask
    raw_shape = zarray['shape']
    raw_offset = zattrs['offset']
    raw_chunks = zarray['chunks']
    raw_resolution = zattrs['resolution']

    array_path = fos.utils.create_zarr(zarr_ds,
                                       array_to_make,
                                       shape=raw_shape,
                                       chunk_size=raw_chunks[0],
                                       offset=raw_offset,
                                       resolution=raw_resolution)

    shutil.copy(
        os.path.join(raw_path, '.zattrs'),
        os.path.join(array_path, '.zattrs'))

    return array_path, raw_shape, raw_offset, raw_chunks, raw_resolution


def make_mask_janelia_crops(zarr_ds: str):

    mask_path, mask_shape, mask_offset, mask_chunks, mask_resolution  \
       = make_empty_like_raw(zarr_ds, 'volumes/mask')

    #  fill it with 255 where there are crops
    mask_arr = read(mask_path)
    mask = da.from_array(mask_arr,
                         chunks=mask_chunks,
                         asarray=True)

    crops_offset, crops_shape = get_crops_coords(zarr_ds)

    for n in range(len(crops_offset)):
        mask[crops_offset[n][0]:crops_offset[n][0]+crops_shape[n][0],
             crops_offset[n][1]:crops_offset[n][1]+crops_shape[n][1],
             crops_offset[n][2]:crops_offset[n][2]+crops_shape[n][2]] = 255

    da.to_zarr(mask,
               mask_path,
               component=None,
               overwrite=True,
               compute=True,
               return_stored=False,
               compressor=Zlib(level=3))


# def crops_to_1_label(zarr_ds: str, organelles: str):
#     for org in organelles:
#         label_path, label_shape, label_offset, label_chunks, label_resolution  \
#            = make_empty_like_raw(zarr_ds, 'volumes/'+str(org))
# 
#         #  fill the crops
#         mask_arr = read(mask_path)
#         mask = da.from_array(mask_arr,
#                              chunks=mask_chunks,
#                              asarray=True)
# 
#         crops_offset, crops_shape = get_crops_coords(zarr_ds)
# 
#         for n in range(len(crops_offset)):
#             mask[crops_offset[n][0]:crops_offset[n][0]+crops_shape[n][0],
#                  crops_offset[n][1]:crops_offset[n][1]+crops_shape[n][1],
#                  crops_offset[n][2]:crops_offset[n][2]+crops_shape[n][2]] = 255
# 
#         da.to_zarr(mask,
#                    mask_path,
#                    component=None,
#                    overwrite=True,
#                    compute=True,
#                    return_stored=False,
#                    compressor=Zlib(level=3))
# 
#         shutil.copy(
#             os.path.join(raw_path, '.zattrs'),
#             os.path.join(mask_path, '.zattrs'))
