"""
Adapted from
https://github.com/neptunes5thmoon/gunpowder/blob/gammaAugment/gunpowder/nodes/gamma_augment.py
"""

import logging
from collections.abc import Iterable
import numpy as np
from gunpowder import BatchFilter

logger = logging.getLogger(__name__)


class GammaAugment(BatchFilter):
    '''

    Args:

        array (:class:`ArrayKey`):

            The intensity array to modify.

        gamma_min (``float``):
        gamma_max (``float``):

    '''

    def __init__(self, arrays, gamma_min, gamma_max):
        if not isinstance(arrays, Iterable):
            arrays = [arrays, ]
        self.arrays = arrays
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        assert self.gamma_max >= self.gamma_min

    def process(self, batch, request):
        sample_gamma_min = (
            max(self.gamma_min, 1. / self.gamma_min) - 1
        ) * (-1)**(self.gamma_min < 1)
        sample_gamma_max = (
            max(self.gamma_max, 1. / self.gamma_max) - 1
        ) * (-1)**(self.gamma_max < 1)
        gamma = np.random.uniform(sample_gamma_min, sample_gamma_max)
        if gamma < 0:
            gamma = 1. / (-gamma + 1)
        else:
            gamma = gamma + 1
        for array in self.arrays:
            raw = batch.arrays[array]
            # raw.attrs['gamma'] = gamma

            assert raw.data.dtype == np.float32 or raw.data.dtype == np.float64, \
                "Gamma augmentation requires float " \
                "types for the raw array (not " + str(
                    raw.data.dtype) + "). Consider using Normalize before."
            assert raw.data.min() >= 0 and raw.data.max() <= 1, \
                "Gamma augmentation expects raw values in [0," \
                "1]. Consider using Normalize before."

            raw.data = raw.data**gamma

            # clip values, we might have pushed them out of [0,1]
            raw.data[raw.data > 1] = 1
            raw.data[raw.data < 0] = 0
