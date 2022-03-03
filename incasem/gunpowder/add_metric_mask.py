from typing import Optional
from scipy import ndimage
import numpy as np
import gunpowder as gp


class AddMetricMask(gp.BatchFilter):
    """Create a mask to ignore predictions at the boundary of objects.

        Args:
            array (gp.ArrayKey):

                An array containing semantic labels or instance labels.

            output_array (gp.ArrayKey):


            dilations (Optional[int]):

                Number of dilations by 1 voxel.

            erosions (Optional[int]):

                Number of erosions by 1 voxel.
    """

    def __init__(
            self,
            array: gp.ArrayKey,
            output_array: gp.ArrayKey,
            dilations: Optional[int] = 4,
            erosions: Optional[int] = 4):

        self.array = array
        self.output_array = output_array
        self.dilations = dilations
        self.erosions = erosions

        self.context = None

    def setup(self):
        self.enable_autoskip()

        spec = self.spec[self.array].copy()
        spec.dtype = np.uint8

        context_size = max(self.dilations, self.erosions)
        self.context = gp.Coordinate(
            (context_size,) * spec.voxel_size.dims()) * spec.voxel_size

        self.provides(self.output_array, spec)

    def prepare(self, request):
        context_roi = request[self.array].roi.grow(self.context, self.context)

        spec = request[self.array].copy()
        spec.roi = context_roi

        deps = gp.BatchRequest()
        deps[self.array] = spec
        return deps

    def process(self, batch, request):
        output = gp.Batch()

        spec = batch[self.array].spec.copy()
        spec.dtype = np.uint8

        labels = batch[self.array].data
        binary_labels = (labels != 0).astype(np.uint8)

        dilated = ndimage.binary_dilation(
            binary_labels, iterations=self.dilations)

        eroded = ndimage.binary_erosion(
            binary_labels, iterations=self.erosions)

        boundary_mask = dilated - eroded
        mask = np.logical_not(boundary_mask).astype(spec.dtype)

        output[self.output_array] = gp.Array(data=mask, spec=spec)

        # Crop back array to the requested size
        output[self.output_array] = output[self.output_array].crop(
            request[self.array].roi)

        return output
