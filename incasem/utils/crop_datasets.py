import logging
from time import time as now

from funlib.persistence import Array, open_ds, prepare_ds
from funlib.geometry import Roi, Coordinate
import daisy

logger = logging.getLogger(__name__)


def crop_dataset_worker(
        block,
        in_ds,
        out_ds,
        read_shift

):
    data = in_ds.to_ndarray(roi=block.read_roi + read_shift, fill_value=0)
    array = Array(data, roi=block.read_roi, voxel_size=in_ds.voxel_size)
    out_ds[block.write_roi] = array[block.write_roi]


def crop_daisy_dataset(
        filename,
        ds_name,
        out_filename,
        out_ds_name,
        crop_offset_voxels,
        crop_shape_voxels,
        chunk_shape,
        dtype,
        num_workers):
    """Crop a daisy dataset.

    Refer to documentation for `crop_datasets`.
    """

    full_ds = open_ds(
        filename,
        ds_name,
        mode='r'
    )

    voxel_size = full_ds.voxel_size

    # If crop shape and offset are not given, copy the full dataset
    if crop_offset_voxels is None:
        crop_offset = full_ds.roi.get_offset()
    else:
        crop_offset = Coordinate(crop_offset_voxels) * voxel_size

    if crop_shape_voxels is None:
        crop_shape = full_ds.roi.get_shape()
    else:
        crop_shape = Coordinate(crop_shape_voxels) * voxel_size

    roi_to_copy = Roi(
        offset=crop_offset,
        shape=crop_shape,
    )
    logger.debug(f"{roi_to_copy=}")

    # Shift both Rois to zero origin to avoid copying artifacts.
    shift_to_origin = Coordinate(
        [min(x, 0) for x in crop_offset]
    )

    # Simple shifting is enough for new dataset.
    # The shift for the existing dataset is carried out in the worker function
    # when the data is actually read from disk.
    roi_to_copy -= shift_to_origin

    logger.debug(f"{roi_to_copy=}")
    logger.debug(f"{full_ds.roi=}")

    out_ds = prepare_ds(
        filename=out_filename,
        ds_name=out_ds_name,
        total_roi=roi_to_copy,
        voxel_size=voxel_size,
        dtype=dtype if dtype else full_ds.dtype,
        write_size=voxel_size * Coordinate(chunk_shape),
        compressor={'id': 'zlib', 'level': 3}
    )

    block_roi = Roi(
        (0,) * voxel_size.dims(),
        Coordinate(chunk_shape) * voxel_size
    )
    start = now()
    task = daisy.Task(
        total_roi=out_ds.roi,
        read_roi=block_roi,
        write_roi=block_roi,
        process_function=lambda block: crop_dataset_worker(
            block,
            in_ds=full_ds,
            out_ds=out_ds,
            read_shift=shift_to_origin,
        ),
        read_write_conflict=False,
        fit='shrink',
        num_workers=num_workers,
        task_id = 'crop_datasets'
    )
    daisy.run_blockwise([task])

    logger.info(f"Done with {out_ds_name} in {now() - start} s")


def crop_datasets(
        filename,
        out_filename,
        datasets,
        out_datasets,
        offset_voxels,
        shape_voxels,
        chunk_shape,
        dtypes,
        num_workers
):
    """Crop or pad multiple datasets in a zarr file.

    A new persistent zarr array is created for each dataset.
    The input datasets are not modified

    Args:

        filename (string):

            Source zarr group.

        out_filename (string):

            Output zarr group.

        datasets (sequence of strings):

            Names of the source datasets.

        out_datasets (sequence of strings):

            Names of the output datasets.

        offset_voxels (sequence of ints):

            Desired offset for all output datasets, in voxels (zyx).
            If not given, read the offset from file.

        shape_voxels (sequence of ints):

            Desired shape for all outputs datasets, in voxels (zyx).
            If not given, read the shape from file.

        chunk_shape (sequence of ints):

            Output chunk shape for all datasets, in voxels (zyx).
            Has to be a multiple of the input chunk shapes.

        dtypes (sequence of strings):

            Numpy dtypes of the output datasets. If not provided,
            the dtypes are copied from the input datasets.

        num_workers (int):

            Number of processes.
    """

    assert len(datasets) == len(out_datasets)
    assert len(datasets) == len(dtypes)

    for ds, out_ds, dt in zip(datasets, out_datasets, dtypes):
        crop_daisy_dataset(
            filename=filename,
            ds_name=ds,
            out_filename=out_filename,
            out_ds_name=out_ds,
            crop_offset_voxels=offset_voxels,
            crop_shape_voxels=shape_voxels,
            chunk_shape=chunk_shape,
            dtype=dt,
            num_workers=num_workers
        )