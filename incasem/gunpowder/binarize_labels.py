import logging
from typing import List

import numpy as np
import gunpowder as gp

logger = logging.getLogger(__name__)


class BinarizeLabels(gp.BatchFilter):
    """Replace all non-zero values with 1 in the given arrays, in-place.

    Output data type changes to ``uint8``.

    Args:

        arrays(``list`` of :class:`ArraysKey`):

            The arrays to binarize.
    """

    def __init__(self, arrays: List[gp.ArrayKey]):

        self.arrays = arrays
        self.in_dtypes = {}

    def setup(self):

        self.enable_autoskip()

        for array in self.arrays:
            spec = self.spec[array].copy()
            self.in_dtypes[array] = spec.dtype

            spec.dtype = np.uint8
            self.updates(array, spec)

        logger.debug(f"{self.in_dtypes=}")

    def prepare(self, request):

        deps = gp.BatchRequest()
        for array in self.arrays:
            if array in request:
                spec = request[array].copy()
                spec.dtype = self.in_dtypes[array]
                logger.debug(f"{spec.dtype=}")
                deps[array] = spec

        return deps

    def process(self, batch, request):

        outputs = gp.Batch()

        for array in self.arrays:
            if array in request:
                spec = batch[array].spec.copy()
                spec.dtype = np.uint8

                binarized = gp.Array(
                    data=(batch[array].data > 0).astype(np.uint8), spec=spec
                )

                outputs[array] = binarized

        return outputs
