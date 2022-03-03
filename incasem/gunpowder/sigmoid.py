import copy
import logging

from scipy.special import expit
import gunpowder as gp

logger = logging.getLogger(__name__)


class Sigmoid(gp.BatchFilter):
    """Apply a softmax operation on the 0th dimension of the array

    Args:
        arrays (gp.ArrayKey):
    """

    def __init__(self, array: gp.ArrayKey):
        self.array = array

    def process(self, batch, request):
        data = batch[self.array].data
        batch[self.array].data = expit(data)
