import logging

import numpy as np
import gunpowder as gp

logger = logging.getLogger(__name__)


class Uint8ToFloat(gp.BatchFilter):
    """Scale uint8 [0,255] to float32 [0,1]

    Args:
        array (gp.ArrayKey):
    """

    def __init__(self, array: gp.ArrayKey):

        self.array = array

    def update(self):

        spec = self.spec[self.array].copy()
        spec.dtype = np.float32
        self.updates(self.array, spec)

    def process(self, batch, request):

        data = batch[self.array].data
        batch[self.array].data = data.astype(np.float32) / 255.0
        batch[self.array].spec.dtype = np.float32
