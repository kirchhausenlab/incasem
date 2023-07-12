import logging
from typing import Dict
import torch

import numpy as np
import gunpowder as gp
from gunpowder.ext import tensorboardX, NoSuchModule

logger = logging.getLogger(__name__)


class ValidationLoss(gp.BatchFilter):
    def __init__(
            self,
            loss,
            inputs: Dict[int, gp.ArrayKey],
            log_dir: str = None,
            log_every: int = 1):

        self.inputs = inputs

        # TODO init cuda only in start() method, similar to gunpowder.torch.Predict
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        self.loss = loss.to(self.device)

        self.log_dir = log_dir
        self.log_every = log_every
        self.iteration = 0

        if not isinstance(tensorboardX, NoSuchModule) and log_dir is not None:
            self.summary_writer = tensorboardX.SummaryWriter(log_dir)
        else:
            self.summary_writer = None
            if log_dir is not None:
                logger.warning(
                    "log_dir given, but tensorboardX is not installed")

    def process(self, batch, request):
        device_loss_inputs = []
        for i in range(len(self.inputs)):
            device_loss_inputs.append(
                torch.from_numpy(batch[self.inputs[i]].data).to(self.device)
            )

        loss = self.loss(*device_loss_inputs)
        batch.loss = loss.detach().cpu().numpy()

        iterable_loss = np.atleast_1d(batch.loss).tolist()
        if self.summary_writer:
            for i, l in enumerate(iterable_loss):
                self.summary_writer.add_scalar(f"loss_{i}", l, self.iteration)

        batch.iteration = self.iteration
        logger.info((
            f'Validation process: iteration={batch.iteration}'
            f' loss={batch.loss}'
        ))

        self.iteration += self.log_every
