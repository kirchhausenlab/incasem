from funlib.persistence import (
    Array,
    ArrayNotFoundError,
    MetaDataFormat,
    get_default_metadata_format,
)
import zarr
from typing import Optional, Sequence, Union


def my_open_ds(
    store,
    mode: Optional[str] = None,
    metadata_format: Optional[MetaDataFormat] = None,
    offset: Optional[Sequence[int]] = None,
    voxel_size: Optional[Sequence[int]] = None,
    axis_names: Optional[Sequence[str]] = None,
    units: Optional[Sequence[str]] = None,
    chunks: Optional[Union[int, Sequence[int], str]] = "strict",
    **kwargs,
) -> Array:
    # Copy the rest of the function body here, with the modification to the zarr.open() call
    # ...
    metadata_format = (
        metadata_format
        if metadata_format is not None
        else get_default_metadata_format()
    )

    try:
        if mode is not None:
            data = zarr.open(store, mode=mode, **kwargs)
        else:
            data = zarr.open(store, **kwargs)
        # data = zarr.open(store, mode=mode, **kwargs)
    except zarr.errors.PathNotFoundError:
        raise ArrayNotFoundError(f"Nothing found at path {store}")

    metadata = metadata_format.parse(
        data.shape,
        data.attrs,
        offset=offset,
        voxel_size=voxel_size,
        axis_names=axis_names,
        units=units,
    )

    return Array(
        data,
        metadata.offset,
        metadata.voxel_size,
        metadata.axis_names,
        metadata.units,
        data.chunks if chunks == "strict" else chunks,
    )


# Then use my_open_ds instead of open_ds in your code
