from typing import List, Tuple
import gunpowder as gp


class SliceArrayByChannels(gp.BatchFilter):
    def __init__(
            self,
            array: gp.ArrayKey,
            output_arrays: List[gp.ArrayKey],
            slices: List[Tuple[int]]):

        self.array = array
        self.output_arrays = output_arrays
        self.slices = slices

        assert len(self.output_arrays) == len(self.slices)

    def setup(self):
        self.enable_autoskip()

        for arr in self.output_arrays:
            if arr == self.array:
                self.updates(arr, self.spec[self.array].copy())
            else:
                self.provides(arr, self.spec[self.array].copy())

    def prepare(self, request):
        deps = gp.BatchRequest()
        for arr in self.output_arrays:
            deps[self.array] = request[arr]

        return deps

    def process(self, batch, request):
        output = gp.Batch()

        for arr, slc in zip(self.output_arrays, self.slices):
            spec = batch[self.array].spec.copy()
            data = batch[self.array].data[slc[0]:slc[1]]

            output[arr] = gp.Array(data=data, spec=spec)

        return output
