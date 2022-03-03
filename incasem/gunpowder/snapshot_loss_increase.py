import logging
import numpy as np
from .snapshot import Snapshot

logger = logging.getLogger(__name__)


class SnapshotLossIncrease(Snapshot):
    def __init__(self, factor, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        self.running_loss = float('inf')

    def write_if(self, batch):
        loss = np.atleast_1d(batch.loss)[0]
        out = self.running_loss * self.factor < loss
        self.running_loss = loss
        return out
