import copy
from typing import List
import logging

import gunpowder as gp

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DeepCopyArrays(gp.BatchFilter):
    """ Deep-copy arrays

    Args:
        arrays (List[gp.ArrayKey]): ArrayKeys to be copied
        output_arrays (List[gp.ArrayKey]): optional, ArrayKeys for outputs
    """

    def __init__(self,
                 arrays: List[gp.ArrayKey],
                 output_arrays: List[gp.ArrayKey] = None):

        self.arrays = arrays

        if output_arrays:
            assert len(arrays) == len(output_arrays)
        self.output_arrays = output_arrays

    def setup(self):

        self.enable_autoskip()

        if self.output_arrays:
            for in_array, out_array in zip(self.arrays, self.output_arrays):
                if not out_array:
                    raise NotImplementedError(
                        'Provide no output_arrays or one for each input_array')
                else:
                    self.provides(out_array, self.spec[in_array].copy())
        else:
            for array in self.arrays:
                self.updates(array, self.spec[array].copy())

    def prepare(self, request):

        deps = gp.BatchRequest()

        if self.output_arrays:
            output_arrays = self.output_arrays
        else:
            output_arrays = self.arrays

        for in_array, out_array in zip(self.arrays, output_arrays):
            deps[in_array] = request[out_array].copy()

        return deps

    def process(self, batch, request):

        outputs = gp.Batch()

        if self.output_arrays:
            output_arrays = self.output_arrays
        else:
            output_arrays = self.arrays

        for in_array, out_array in zip(self.arrays, output_arrays):
            outputs[out_array] = copy.deepcopy(batch[in_array])
            outputs[out_array].data = batch[in_array].data.copy()

        return outputs
