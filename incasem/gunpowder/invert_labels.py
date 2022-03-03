from typing import Optional, List
import numpy as np
import gunpowder as gp


class InvertLabels(gp.BatchFilter):
    def __init__(self, arrays: List[gp.ArrayKey]):
        """InvertLabels.

        Replace all zeros with ones, non-zero values with zero
        in the given arrays, in-place

        Args:
            arrays:
        """

        self.arrays = arrays

    def setup(self):
        self.enable_autoskip()

        for array in self.arrays:
            spec = self.spec[array].copy()
            spec.dtype = np.uint8
            self.updates(array, spec)

    def prepare(self, request):
        deps = gp.BatchRequest()
        for array in self.arrays:
            deps[array] = request[array].copy()

        return deps

    def process(self, batch, request):
        outputs = gp.Batch()
        for array in self.arrays:
            spec = batch[array].spec.copy()
            spec.dtype = np.uint8

            inverted = gp.Array(
                data=(batch[array].data == 0).astype(np.uint8), spec=spec
            )

            outputs[array] = inverted

        return outputs
