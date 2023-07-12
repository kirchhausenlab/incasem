from typing import Optional
import numpy as np
from skimage import morphology
import gunpowder as gp


class AddBoundaryLabels(gp.BatchFilter):
    def __init__(
            self,
            array: gp.ArrayKey,
            output_array: gp.ArrayKey,
            dtype: Optional[str] = 'uint8',
            thickness: Optional[int] = 4):

        self.array = array
        self.output_array = output_array
        self.dtype = dtype
        self.thickness = thickness

    def setup(self):
        self.enable_autoskip()

        spec = self.spec[self.array].copy()
        spec.dtype = np.dtype(self.dtype)
        spec.interpolatable = False
        self.provides(self.output_array, spec)

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.array] = request[self.output_array]
        return deps

    def process(self, batch, request):
        output = gp.Batch()

        spec = self.spec[self.output_array].copy()
        spec.roi = batch[self.array].spec.roi

        # dilate
        dilated = morphology.binary_dilation(
            image=batch[self.array].data,
            selem=morphology.ball(self.thickness)
        ).astype(self.dtype)

        # # erode
        # eroded = morphology.binary_erosion(
        # image=batch[self.array].data,
        # selem=morphology.ball(2)
        # ).astype(self.dtype)

        # subtract
        boundary = dilated - batch[self.array].data

        output[self.output_array] = gp.Array(data=boundary, spec=spec)
        return output
