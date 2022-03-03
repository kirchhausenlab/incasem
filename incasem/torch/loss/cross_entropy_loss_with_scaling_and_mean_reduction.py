import logging
import torch

logger = logging.getLogger(__name__)


class CrossEntropyLossWithScalingAndMeanReduction(torch.nn.Module):
    def __init__(self, weight=None, device='cuda'):
        super().__init__()

        self.register_buffer('weight', weight)
        self.device = device
        logger.debug(f'Using device {self.device}')

        cross_entropy = torch.nn.CrossEntropyLoss(
            reduction='none',
            weight=self.weight
        )
        self.add_module('cross_entropy', cross_entropy)

    def forward(self, input, target, mask=None, scaling=None):

        # logger.debug(f'{input.shape=}')
        # logger.debug(f'{target.shape=}')
        # logger.debug(f'{scaling.shape=}')
        loss_per_elem = self.cross_entropy(
            input=input,
            target=target,
        )

        logger.debug(f'{float(input.sum())=}')
        logger.debug(f'{float(target.sum())=}')
        logger.debug(f'Loss sum={float(loss_per_elem.sum())}')

        # Masking
        if mask is not None:
            assert loss_per_elem.shape == mask.shape
            loss_per_elem = loss_per_elem * mask

        # Scaling
        if scaling is None:
            logger.warning(f'No scaling argument passed.')
            scaling = torch.ones_like(loss_per_elem, device=self.device)

        logger.debug(f'{loss_per_elem.shape=}')
        assert loss_per_elem.shape == scaling.shape

        loss_per_elem = loss_per_elem * scaling
        logger.debug(f'{loss_per_elem.shape=}')
        logger.debug(
            f'Scaled loss per elem sum={float(loss_per_elem.sum())}')

        loss_reduced = loss_per_elem.sum() / scaling.sum()
        logger.debug(f'{loss_reduced.shape=}')

        return loss_reduced
