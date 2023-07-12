import logging
import os
import json


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
            crop_path = os.path.join(crop_folder, crop)
            if not os.path.isdir(crop_path):
                continue

            config[crop] = {}
            with open(os.path.join(crop_path, '.zattrs')) as f:
                zattrs = json.load(f)  # offset and resolution
            with open(os.path.join(crop_path, '.zarray')) as f:
                zarray = json.load(f)  # shape and voxel size

            config[crop]['file'] = zarr_path_to_data
            config[crop]['offset'] = zattrs['offset']
            config[crop]['shape'] = zarray['shape']
            config[crop]['voxel_size'] = zattrs['resolution']
            config[crop]['raw'] = 'volumes/raw_equalized_0.02'
            config[crop]['labels'] = {'volumes/labels/'+org+'/'+crop: 1}

        config_path = os.path.join(
                        os.path.dirname(zarr_ds),
                        org+'_config.json')
        with open(config_path, 'w') as outfile:
            json.dump(config, outfile, indent=4, sort_keys=True)

        logger.info(f'created: {config_path}')
