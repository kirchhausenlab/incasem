import numpy as np
import gunpowder as gp


class MergeArrays(gp.BatchFilter):
    def __init__(self, arrays, output_array):

        assert len(arrays) > 1
        self.arrays = arrays
        self.output_array = output_array

    def setup(self):

        self.enable_autoskip()

        # TODO fix this
        # for a, b in zip(self.arrays, self.arrays[1:]):
            # assert self.spec[a] == self.spec[b]

        spec = self.spec[self.arrays[0]].copy()
        self.provides(self.output_array, spec)

    def prepare(self, request):

        deps = gp.BatchRequest()
        for array in self.arrays:
            deps[array] = request[self.output_array]

        return deps

    def process(self, batch, request):

        output = gp.Batch()
        spec = batch[self.arrays[0]].spec.copy()

        # TODO handle keys not present in batch
        arrays = [batch[key].data for key in self.arrays]
        data = np.stack(arrays)

        output[self.output_array] = gp.Array(data=data, spec=spec)

        return output
