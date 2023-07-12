from typing import List
import numpy as np
import gunpowder as gp


class MergeMasks(gp.BatchFilter):
    def __init__(
            self,
            arrays: List[gp.ArrayKey],
            output_array: gp.ArrayKey):
        """Merge multiple binary masks with a logical and

        Args:
            arrays: list of binary masks for different structures
            output_array:
        """

        self.arrays = arrays
        self.output_array = output_array

    def setup(self):
        self.enable_autoskip()

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

        mask = np.logical_and.reduce(
            [batch[key].data.astype(bool) for key in self.arrays]
        )

        mask = mask.astype(np.uint8)

        output[self.output_array] = gp.Array(data=mask, spec=spec)
        return output
