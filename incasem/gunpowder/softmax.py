import copy
import logging

import numpy as np
import scipy
import gunpowder as gp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Softmax(gp.BatchFilter):
    """Apply a softmax operation on the 0th dimension of the array

    Args:
        arrays (gp.ArrayKey):
    """

    def __init__(self, array: gp.ArrayKey, output_array: gp. ArrayKey = None):
        self.array = array
        self.output_array = output_array

    def setup(self):
        self.enable_autoskip()

        if self.output_array:
            self.provides(self.output_array, self.spec[self.array].copy())
        else:
            self.updates(self.array, self.spec[self.array].copy())

    def prepare(self, request):
        deps = gp.BatchRequest()
        if self.output_array:
            deps[self.array] = request[self.output_array].copy()
        else:
            deps[self.array] = request[self.array].copy()

        return deps

    def process(self, batch, request):
        assert batch[self.array].data.ndim == 4, \
            (f'Softmax only implemented for 4-dimensional input. '
             f'{self.array} is {batch[self.array].data.ndim} dimensional')

        outputs = gp.Batch()

        if self.output_array:
            out_array = self.output_array
        else:
            out_array = self.array

        outputs[out_array] = copy.deepcopy(batch[self.array])
        outputs[out_array].data = \
            scipy.special.softmax(batch[self.array].data, axis=0)

        return outputs
