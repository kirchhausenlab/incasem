import logging
import numpy as np

from gunpowder import BatchRequest
from gunpowder import BatchFilter

logger = logging.getLogger(__name__)


class MaskedIntensityAugment(BatchFilter):
    '''Randomly scale and shift the values of a masked intensity array.

    Args:

        array (:class:`ArrayKey`):

            The intensity array to modify.

        scale_min (``float``):
        scale_max (``float``):
        shift_min (``float``):
        shift_max (``float``):

            The min and max of the uniformly randomly drawn scaling and
            shifting values for the intensity augmentation. Intensities are
            changed as::

                a = a * scale + shift

        mask (:class:`ArrayKey`):

            Perform the augmentation only on the voxels with non-zero value of
            a mask.
    '''

    def __init__(
            self,
            array,
            scale_min,
            scale_max,
            shift_min,
            shift_max,
            mask):
        self.array = array
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.shift_min = shift_min
        self.shift_max = shift_max
        self.mask = mask

    def setup(self):
        self.enable_autoskip()
        self.updates(self.array, self.spec[self.array])

    def prepare(self, request):
        deps = BatchRequest()
        deps[self.array] = request[self.array].copy()
        if self.mask is not None:
            deps[self.mask] = request[self.array].copy()
        return deps

    def process(self, batch, request):

        raw = batch.arrays[self.array]

        assert raw.data.dtype == np.float32 or raw.data.dtype == np.float64, \
            "Intensity augmentation requires float types for the raw array" \
            " (not " + str(raw.data.dtype) + "). Consider using Normalize before."
        assert raw.data.min() >= 0 and raw.data.max() <= 1, \
            "Intensity augmentation expects raw values in [0,1]. "\
            "Consider using Normalize before."

        assert raw.data.shape == batch[self.mask].data.shape, \
            "Intensity augmentation expects the mask to be of the same " \
            "shape as the raw data."

        scale = np.random.uniform(low=self.scale_min, high=self.scale_max)
        shift = np.random.uniform(low=self.shift_min, high=self.shift_max)

        binary_mask = batch[self.mask].data.astype(bool)
        scale = binary_mask * scale + \
            np.logical_not(binary_mask).astype(raw.data.dtype)
        shift = binary_mask * shift

        raw.data = self.__augment(raw.data, scale, shift)

        # clip values, we might have pushed them out of [0,1]
        raw.data[raw.data > 1] = 1
        raw.data[raw.data < 0] = 0

    def __augment(self, a, scale, shift):
        # Custom choice: No mean subtraction to augment more on bright values
        return a * scale + shift
