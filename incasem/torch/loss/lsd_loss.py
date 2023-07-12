import logging
import time
import torch


logger = logging.getLogger(__name__)


class LsdLoss(torch.nn.Module):
    def __init__(self, weight=None, device='cuda'):
        super().__init__()

        self.register_buffer('weight', weight)
        self.device = device
        logger.debug(f'Using device {self.device}')

        task_loss = torch.nn.CrossEntropyLoss(
            reduction='none',
            weight=self.weight
        )
        self.add_module('task_loss', task_loss)

        lsd_loss = torch.nn.MSELoss(reduction='none')
        self.add_module('lsd_loss', lsd_loss)

    def forward(
            self,
            input_task,
            input_lsd,
            target_task,
            target_lsd,
            mask=None,
            scaling=None):
        """forward.

        Args:
            input:
            target:
            mask:
            scaling:
        """

        start = time.time()

        # logger.debug(f'{input.shape=}')
        # logger.debug(f'{target.shape=}')
        # logger.debug(f'{scaling.shape=}')

        # Convert target lsds to [0,1] float
        target_lsd = target_lsd * (1.0 / 255)

        lsd_loss_per_elem = self.lsd_loss(input_lsd, target_lsd)
        lsd_loss_per_voxel = torch.mean(lsd_loss_per_elem, dim=1)

        # static weighting of LSDs
        weight_per_voxel = self.weight[target_task]
        lsd_loss_per_voxel *= weight_per_voxel

        loss_per_voxel = self.task_loss(
            input=input_task,
            target=target_task,
        )

        assert lsd_loss_per_voxel.shape == loss_per_voxel.shape
        # logger.debug(f'{float(input.sum())=}')
        # logger.debug(f'{float(target.sum())=}')
        logger.debug(f'Loss sum={float(loss_per_voxel.sum())}')

        # Masking
        if mask is not None:
            assert loss_per_voxel.shape == mask.shape
            loss_per_voxel = loss_per_voxel * mask

            assert lsd_loss_per_voxel.shape == mask.shape
            lsd_loss_per_voxel = lsd_loss_per_voxel * mask

        # Scaling
        if scaling is None:
            logger.warning(f'No scaling argument passed.')
            scaling = torch.ones_like(loss_per_voxel, device=self.device)

        logger.debug(f'{loss_per_voxel.shape=}')
        logger.debug(f"{scaling.shape=}")
        assert loss_per_voxel.shape == scaling.shape

        loss_per_voxel = loss_per_voxel * scaling
        logger.debug(f'{loss_per_voxel.shape=}')
        logger.debug(
            f'Scaled loss per elem sum={float(loss_per_voxel.sum())}')

        # LSD Scaling
        lsd_loss_per_voxel = lsd_loss_per_voxel * scaling

        loss_reduced = loss_per_voxel.sum() / scaling.sum()
        logger.debug(f'{loss_reduced.shape=}')

        lsd_loss_reduced = lsd_loss_per_voxel.sum() / scaling.sum()
        logger.debug(f'{lsd_loss_reduced.shape=}')

        loss_total = loss_reduced + lsd_loss_reduced

        # self.components = (loss_reduced, lsd_loss_reduced)
        logger.debug(f"LSD loss in {time.time() - start:.6f} s")

        return torch.cat([
            loss_total.reshape(1),
            loss_reduced.reshape(1),
            lsd_loss_reduced.reshape(1),
        ])
