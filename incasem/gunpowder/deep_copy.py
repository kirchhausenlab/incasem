import copy
from typing import List
import logging

import gunpowder as gp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DeepCopy(gp.BatchFilter):
    """ deep copy arrays to ensure that they are contiguous in memory

    Args:
        arrays (List[gp.ArrayKey]): ArrayKeys for arrays to be copied
    """

    def __init__(self, arrays: List[gp.ArrayKey]):
        self.arrays = arrays

    # copy the specs because everything is copied here
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
            outputs[array].data = batch[array].data.copy()
        return outputs
