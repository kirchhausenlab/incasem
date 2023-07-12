import copy
from typing import List
import logging

import numpy as np
import gunpowder as gp

logger = logging.getLogger(__name__)


class Squeeze(gp.BatchFilter):
    """Squeeze a batch at a given axis

    Args:
        arrays (List[gp.ArrayKey]): ArrayKeys to squeeze.
        axis: Position of the single-dimensional axis to remove, defaults to 0.
    """

    def __init__(self, arrays: List[gp.ArrayKey], axis: int = 0):
        self.arrays = arrays
        self.axis = axis

        if self.axis != 0:
            raise NotImplementedError(
                'Squeeze only supported for leading dimension')

    def setup(self):
        self.enable_autoskip()
        for array in self.arrays:
            self.updates(array, self.spec[array].copy())

    def prepare(self, request):
        deps = gp.BatchRequest()
        for array in self.arrays:
            deps[array] = request[array].copy()
        return deps

    def process(self, batch, request):
        outputs = gp.Batch()
        for array in self.arrays:
            outputs[array] = copy.deepcopy(batch[array])
            outputs[array].data = np.squeeze(batch[array].data, self.axis)
            logger.debug(f'{array} shape: {outputs[array].data.shape}')

        return outputs
