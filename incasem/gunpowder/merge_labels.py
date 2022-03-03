from typing import Dict, List, Optional
import numpy as np
import gunpowder as gp


class MergeLabels(gp.BatchFilter):
    """Merge the labels from all input arrays into the output array

    Multiply each array with the given integer value.
    If there are multiple labels assigned for a voxel
    we resolve with the specified strategy.

    Args:

        classes:

            dictionary of foreground label classes

        output_array:

        dtype:

        ambiguous_labels (str):
            `max` or `background` or `preference`.
    """

    def __init__(
            self,
            classes: Dict[gp.ArrayKey, int],
            output_array: gp.ArrayKey,
            dtype: Optional[str] = 'uint32',
            ambiguous_labels: Optional[str] = 'background'):

        self.classes = classes
        self.num_classes = len(self.classes) + 1
        self.output_array = output_array
        self.dtype = dtype
        self.ambiguous_labels = ambiguous_labels

        assert self.ambiguous_labels in (
            'max', 'background', 'preference')

    def setup(self):

        self.enable_autoskip()

        spec = self.spec[list(self.classes.keys())[0]].copy()
        spec.dtype = np.dtype(self.dtype)
        self.provides(self.output_array, spec)

    def prepare(self, request):

        deps = gp.BatchRequest()
        for array, _ in self.classes.items():
            spec = request[self.output_array].copy()
            spec.dtype = self.spec[array].dtype
            deps[array] = spec

        return deps

    def process(self, batch, request):

        output = gp.Batch()
        spec = batch[list(self.classes.keys())[0]].spec.copy()
        spec.dtype = np.dtype(self.dtype)

        if self.ambiguous_labels == 'max':
            labels = np.max(
                np.array(
                    [batch[array].data * class_id for array,
                     class_id in self.classes.items()]),
                axis=0
            ).astype(self.dtype)

        elif self.ambiguous_labels == 'background':
            sum_of_binaries = np.sum(
                np.array(
                    [batch[array].data for array, _ in self.classes.items()]),
                axis=0
            )
            ambiguous = sum_of_binaries > 1

            labels = np.sum(
                np.array(
                    [batch[array].data * class_id for array,
                     class_id in self.classes.items()]),
                axis=0
            ).astype(self.dtype)

            labels_mask = labels > 0

            not_ambiguous = np.logical_and(
                labels_mask, np.logical_not(ambiguous))

            labels *= not_ambiguous.astype(self.dtype)

        elif self.ambiguous_labels == 'preference':
            raise NotImplementedError((
                'Resolving ambiguous labels with a custom preference'
                ' not implemented yet.'
            ))
        else:
            raise ValueError(
                f"Invalid ambiguous labels strategy {self.ambiguous_labels}")

        output[self.output_array] = gp.Array(data=labels, spec=spec)
        return output
