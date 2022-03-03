import logging
from typing import Dict
import torch

import gunpowder as gp

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CountIteration(gp.BatchFilter):
    def __init__(self):
        self.iteration = 0

    def setup(self):
        pass

    def prepare(self, request):
        return None

    def process(self, batch, request):
        # TODO copy the batch?
        logger.debug(f'Incoming iteration: {batch.iteration}')

        batch.iteration = self.iteration
        logger.info((
            f'Iteration={self.iteration}'
        ))
        self.iteration += 1
