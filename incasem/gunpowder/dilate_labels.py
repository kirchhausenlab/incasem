from typing import Optional
from skimage import morphology
import gunpowder as gp


class DilateLabels(gp.BatchFilter):
    """DilateLabels.

        Args:
            array (gp.ArrayKey):
                An array containing binary labels
            thickness (Optional[int]):
                Radius of the sphere that is used for dilation
    """

    def __init__(
            self,
            array: gp.ArrayKey,
            thickness: Optional[int] = 3):

        self.array = array
        self.thickness = thickness

    def setup(self):
        self.enable_autoskip()

        spec = self.spec[self.array].copy()
        self.updates(self.array, spec)

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.array] = request[self.array]
        return deps

    def process(self, batch, request):
        output = gp.Batch()

        spec = batch[self.array].spec.copy()

        # dilate
        dilated = morphology.binary_dilation(
            image=batch[self.array].data,
            selem=morphology.ball(self.thickness)
        ).astype(spec.dtype)

        output[self.array] = gp.Array(data=dilated, spec=spec)
        return output
