"""Helpers for parsing recurring command line arguments"""


def split_zarr_path(path):
    if path is None:
        return None
    filename, extension, ds_name = path.rpartition('.zarr/')
    filename = (filename + extension).rstrip('/')
    ds_name = ds_name.rstrip('/')
    return filename, ds_name
