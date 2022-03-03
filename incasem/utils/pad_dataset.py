import logging
import os
import shutil
import json
from operator import add

from fibsem_tools.io import read
import dask.array as da

from numcodecs import Zlib
import incasem as fos


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def pad_dataset(zarr: str,
                ds: str,
                pad: list,
                chunks: list,
                workers=20,
                threads_per_worker=3):

    ds_path = os.path.join(zarr, ds)

    padded_ds = ds + '_padded'
    padded_path = os.path.join(zarr, padded_ds)
    if os.path.exists(padded_path) and os.path.isdir(padded_path):
        shutil.rmtree(padded_path)

    # client = fos.utils.init_client(workers, threads_per_worker)
    # dask_address = client.scheduler_info()["services"]["dashboard"]
    # logger.info(f'\n\nDask dashboard at: http://localhost:{dask_address}\n\n')

    # make empty array
    with open(os.path.join(ds_path, '.zattrs')) as f:
        zattrs = json.load(f)
    with open(os.path.join(ds_path, '.zarray')) as f:
        zarray = json.load(f)

    #  create empty mask
    nm_per_voxel = zattrs['resolution']
    init_shape = [int(s) for s in zarray['shape']]
    padded_shape = \
        [init_shape[0] + pad[0] + pad[1],
         init_shape[1] + pad[2] + pad[3],
         init_shape[2] + pad[4] + pad[5]]

    init_offset = [int(s) for s in zattrs['offset']]
    padded_offset = \
        [init_offset[0] - pad[0] * nm_per_voxel[0],
         init_offset[1] - pad[2] * nm_per_voxel[1],
         init_offset[2] - pad[4] * nm_per_voxel[2]]

    padded_chunks = zarray['chunks']
    padded_resolution = zattrs['resolution']

    padded_path = fos.utils.create_zarr(zarr,
                                        padded_ds,
                                        shape=padded_shape,
                                        chunk_size=padded_chunks[0],
                                        offset=padded_offset,
                                        resolution=padded_resolution)

    logger.debug(f'{padded_path=}')
    pad_zattrs = zattrs
    pad_zattrs['offset'] = padded_offset

    pad_zattrs_path = os.path.join(padded_path, '.zattrs')
    with open(pad_zattrs_path, 'w') as f:
        json.dump(pad_zattrs, f)

    #  fill it with 255 where there are crops
    padded = read(padded_path)
    padded_data = da.from_array(padded,
                                chunks=chunks,
                                asarray=True)

    original = read(ds_path)
    original_data = da.from_array(original,
                                  chunks=chunks,
                                  asarray=True)

    pad_to_apply = ((pad[0], pad[1]),  # pad before and after this axis
                    (pad[2], pad[3]),
                    (pad[4], pad[5]))

    logger.info(f'{pad_to_apply=}')
    padded_data = da.pad(original_data, pad_to_apply, mode='edge')
    logger.debug(f'{padded=}')
    logger.debug(f'{padded_data=}')

    padded_data = da.rechunk(padded_data)

    da.to_zarr(padded_data,
               padded_path,
               component=None,
               overwrite=True,
               compute=True,
               return_stored=False,
               compressor=Zlib(level=3))

    pad_zattrs_path = os.path.join(padded_path, '.zattrs')
    with open(pad_zattrs_path, 'w') as f:
        json.dump(pad_zattrs, f)


