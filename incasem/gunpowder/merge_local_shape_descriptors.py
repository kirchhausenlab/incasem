from typing import List, Optional
import numpy as np
import gunpowder as gp


class MergeLocalShapeDescriptors(gp.BatchFilter):
    def __init__(
            self,
            arrays: List[gp.ArrayKey],
            output_array: gp.ArrayKey,
            ambiguous: Optional[str] = 'max'):
        """Merge the local shape descriptors from all input arrays

        If there are multiple labels assigned for a voxel
        we resolve with the specified strategy.

        Args:
            arrays: list of shape descriptors for different structures
            output_array:
            ambiguous_labels (str):
                `max` or `background` or `preference`.
        """

        self.arrays = arrays
        self.output_array = output_array
        self.ambiguous = ambiguous

        assert self.ambiguous in ('max', 'background', 'preference')

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

        if self.ambiguous == 'max':
            lsds = np.max(
                np.array(
                    [batch[array].data for array in self.arrays]),
                axis=0
            )

        elif self.ambiguous == 'background':
            sum_of_binaries = np.sum(
                np.array(
                    [np.any(batch[array].data.astype(bool), axis=0)
                     for array in self.arrays]
                ),
                axis=0
            )
            ambiguous = sum_of_binaries > 1

            lsds = np.sum(
                np.array(
                    [batch[array].data for array in self.arrays]
                ),
                axis=0
            )

            mask = np.any(lsds.astype(bool), axis=0)

            not_ambiguous = np.logical_and(
                mask, np.logical_not(ambiguous))

            lsds *= not_ambiguous.astype(spec.dtype)

        elif self.ambiguous == 'preference':
            raise NotImplementedError((
                'Resolving ambiguous labels with a custom preference'
                ' not implemented yet.'
            ))
        else:
            raise ValueError(
                f"Invalid ambiguous labels strategy {self.ambiguous}")

        output[self.output_array] = gp.Array(data=lsds, spec=spec)
        return output
