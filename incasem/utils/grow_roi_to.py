import logging
import gunpowder as gp

logger = logging.getLogger(__name__)


def grow_roi_to(roi, target_shape, voxel_size):
    grow_size_total = target_shape - roi.get_shape()

    # only grow if the spec roi is smaller than the target roi
    grow_size_total = gp.Coordinate(
        (max(g, 0) for g in grow_size_total)
    )

    grow_size_neg = (grow_size_total // voxel_size) // 2
    grow_size_pos = ((grow_size_total + voxel_size)
                     // voxel_size) // 2

    output_roi = roi.grow(
        grow_size_neg * voxel_size,
        grow_size_pos * voxel_size
    )

    return output_roi
