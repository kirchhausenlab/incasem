import logging
import torch

from .generalized_dice_loss import GeneralizedDiceLoss

logger = logging.getLogger(__name__)


class CrossEntropyAndDiceLoss(torch.nn.Module):
    def __init__(self, num_classes, weight=None):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.debug(f'Using device {self.device}')

        self._num_classes = num_classes

        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=weight)
        self.dice = GeneralizedDiceLoss(num_classes=num_classes)

    def forward(self, input, target):
        return self.cross_entropy.forward(input, target) + \
            self.dice(input, target)

    @property
    def num_classes(self):
        return self._num_classes
