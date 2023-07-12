import copy
from typing import List
import logging

import numpy as np
import gunpowder as gp

logger = logging.getLogger(__name__)


class ToDtype(gp.BatchFilter):
    """ Cast arrays to another numerical datatype

    Args:
        arrays (List[gp.ArrayKey]): ArrayKeys for typecasting
        dtype: output data type as string
        output_arrays (List[gp.ArrayKey]): optional, ArrayKeys for outputs
    """

    def __init__(self,
                 arrays: List[gp.ArrayKey],
                 dtype,
                 output_arrays: List[gp.ArrayKey] = None):
        self.arrays = arrays
        self.dtype = np.dtype(dtype)

        if output_arrays:
            if len(arrays) != len(output_arrays):
                raise NotImplementedError((
                    'Provide no output_arrays at all, '
                    'or one for each input_array.'
                ))
        self.output_arrays = output_arrays

    def setup(self):
        self.enable_autoskip()

        if self.output_arrays:
            for in_array, out_array in zip(self.arrays, self.output_arrays):
                spec = self.spec[in_array].copy()
                spec.dtype = self.dtype
                self.provides(out_array, spec)
        else:
            for array in self.arrays:
                spec = self.spec[array].copy()
                spec.dtype = self.dtype
                self.updates(array, spec)

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
            logger.debug((
                f'{type(self).__name__} upstream provider spec dtype: '
                f'{outputs[in_array].spec.dtype}'
            ))
            outputs[out_array].spec.dtype = self.dtype
            outputs[out_array].data = batch[in_array].data.astype(self.dtype)

        return outputs
