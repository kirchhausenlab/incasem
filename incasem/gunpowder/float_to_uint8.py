import logging

import numpy as np
import gunpowder as gp
from skimage.util import img_as_ubyte

logger = logging.getLogger(__name__)


class FloatToUint8(gp.BatchFilter):
    """Scale from [0,1] to uint8 [0,255]

    Args:
        array (gp.ArrayKey):
    """

    def __init__(self, array: gp.ArrayKey):
        self.array = array

    def process(self, batch, request):
        data_float = batch[self.array].data
        batch[self.array].data = img_as_ubyte(data_float)
        batch[self.array].spec.dtype = np.uint8
