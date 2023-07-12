"""Iteratively dilate instance labels or semantic labels by a voxel."""
import numpy as np
from scipy import ndimage


def dilate_labels(
        labels,
        iterations,
        dtype,
        ambiguous_labels='max',
        background=0):

    max_label = np.max(labels)
    dilated = np.zeros(shape=labels.shape, dtype=dtype)

    for lab in np.unique(labels):
        if lab == background:
            continue

        mask = labels == lab

        dilated_mask = ndimage.binary_dilation(
            mask, iterations=iterations).astype(dtype)

        dilated_instance = dilated_mask * lab
        dilated = __merge_labels(
            dilated_instance,
            dilated,
            ambiguous_labels,
            max_label
        )

    return dilated


def __merge_labels(source, target, ambiguous_labels, max_label):
    if ambiguous_labels == 'max':
        target = np.maximum(source, target)

    elif ambiguous_labels == 'preserve':
        raise NotImplementedError(
            'Leaving ambiguous labels unchanged not implemented yet')

    else:
        raise ValueError(
            f"Invalid ambiguous labels strategy {ambiguous_labels}")

    return target
