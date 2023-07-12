import copy
import logging

import gunpowder as gp

logger = logging.getLogger(__name__)


class PickChannel(gp.BatchFilter):
    """Pick only one channel in the 0th dimension of a 4-dimensional array

    Args:
        array (gp.ArrayKey):
        output_array (gp.ArrayKey):
        channel(int): the channel in the 0th dimension to be kept
    """

    def __init__(
            self,
            array: gp.ArrayKey,
            channel: int,
            output_array: gp.ArrayKey = None):
        self.array = array
        self.channel = channel
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
            (f'PickChannel only implemented for 4-dimensional input. '
             f'{self.array} is {batch[self.array].data.ndim} dimensional')

        outputs = gp.Batch()
        if self.output_array:
            out_array = self.output_array
        else:
            out_array = self.array

        outputs[out_array] = copy.deepcopy(batch[self.array])
        outputs[out_array].data = batch[self.array].data[self.channel, :, :, :]

        return outputs
