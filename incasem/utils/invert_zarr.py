import logging
import os
import shutil

from fibsem_tools.io import read

import dask.array as da

from numcodecs import Zlib

import incasem as fos


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def invert_zarr(zarr_ds: str,
                chunks: list,
                workers=20,
                threads_per_worker=3):

    inverted_raw_path = os.path.join(zarr_ds, 'volumes', 'raw_inverted')
    if os.path.exists(inverted_raw_path) and os.path.isdir(inverted_raw_path):
        shutil.rmtree(inverted_raw_path)

    # client = fos.utils.init_client(workers, threads_per_worker)
    # dask_address = client.scheduler_info()["services"]["dashboard"]
    # logger.info(f'\n\nDask dashboard at: http://localhost:{dask_address}\n\n')

    inv_path, _, _, chunks, _ = \
        fos.utils.make_empty_like_raw(zarr_ds, 'volumes/raw_inverted')

    #  fill it with 255 where there are crops
    inv = read(inv_path)
    inv_data = da.from_array(inv,
                             chunks=chunks,
                             asarray=True)

    raw = read(os.path.join(zarr_ds, 'volumes/raw'))
    raw_data = da.from_array(raw,
                             chunks=chunks,
                             asarray=True)

    inv_data[:, :, :] = 255 - raw_data[:, :, :]

    da.to_zarr(inv_data,
               inverted_raw_path,
               component=None,
               overwrite=True,
               compute=True,
               return_stored=False,
               compressor=Zlib(level=3))

    shutil.copy(
            os.path.join(zarr_ds, 'volumes/raw/.zattrs'),
            os.path.join(zarr_ds, 'volumes/raw_inverted/.zattrs'))
