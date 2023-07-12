import logging
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CrossEntropyLossDebug(torch.nn.CrossEntropyLoss):
    # def __init__(self):
    # super().__init__(reduction="none")

    def forward(self, input, target):
        logger.debug(f'{float(input.sum())=}')
        logger.debug(f'{float(target.sum())=}')
        return torch.nn.functional.cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction)

        # return torch.mean(loss)
