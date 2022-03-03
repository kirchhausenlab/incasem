import logging
import numpy as np
from incasem.utils import equalize_adapthist
import gunpowder as gp

logger = logging.getLogger(__name__)


class IntensityEqualization(gp.BatchFilter):
    """Perform Contrast Limited Adaptive Histogram Equalization (CLAHE)
    on the entire ``array``.

    Args:

        array (:class:`ArrayKey`):

            The key of the array to modify.

        kernel_size (``int`` or ``tuple`` of ``int``):

            Histogram is adapted for each block with this size.
            Additionally, the request will be modified to include context
            of half of this size.

        clip_limit (``float``):

            Histogram is clipped at this normalized limit before equalization.

    """

    def __init__(self, array, kernel_size=32, clip_limit=0.01):
        self.array = array
        self.kernel_size = kernel_size
        self.clip_limit = clip_limit

        self.context = None

    def setup(self):
        spec = self.spec[self.array].copy()
        spec.dtype = np.uint8
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size,) * spec.voxel_size.dims()

        # Context is half of the kernel_sized, rounded up
        self.context = (gp.Coordinate(self.kernel_size) +
                        (1,) * spec.voxel_size.dims()) / 2
        self.context *= spec.voxel_size

        if spec.roi is not None:
            spec.roi = spec.roi.grow(-self.context, -self.context)

        self.updates(self.array, spec)

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

        data = batch[self.array].data
        equalized = equalize_adapthist(
            image=data,
            kernel_size=self.kernel_size,
            clip_limit=self.clip_limit
        )
        equalized = gp.Array(data=equalized, spec=spec)

        # Crop back array to the requested size
        output[self.array] = equalized.crop(request[self.array].roi)

        return output
