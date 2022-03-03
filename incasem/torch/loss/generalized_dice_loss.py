import logging
from pytorch3dunet.unet3d.losses import \
    GeneralizedDiceLoss as GeneralizedDiceLossOneHot
from pytorch3dunet.unet3d.utils import expand_as_one_hot

logger = logging.getLogger(__name__)


class GeneralizedDiceLoss(GeneralizedDiceLossOneHot):
    def __init__(self, num_classes, sigmoid_normalization=True, epsilon=1e-6):
        super().__init__(
            sigmoid_normalization=sigmoid_normalization,
            epsilon=epsilon
        )
        self._num_classes = num_classes

    def dice(self, input, target, weight):
        # TODO introduce additional asserts here
        target = expand_as_one_hot(target, C=self._num_classes)
        return super().dice(input, target, weight)

    @property
    def num_classes(self):
        return self._num_classes
