import logging
import numpy as np
import gunpowder as gp

logger = logging.getLogger(__name__)


class SaveBlockPosition(gp.BatchFilter):
    """Save offset and shape of an array into another non-spatial array.

    Args:

        raw (:class:`ArrayKey`):

            Data :class:`ArrayKey`, whose coordinates we log.

        log (:class:`ArrayKey`):

            Non-spatial :class:`ArrayKey` to store the coordinates.
    """

    def __init__(self, raw: gp.ArrayKey, log: gp.ArrayKey):

        self.raw = raw
        self.log = log

    def setup(self):

        self.enable_autoskip()
        self.provides(
            self.log,
            gp.ArraySpec(nonspatial=True)
        )

    def prepare(self, request):

        deps = gp.BatchRequest()
        deps[self.raw] = request[self.log].copy()
        return deps

    def process(self, batch, request):

        log = np.zeros((2, 3), dtype=int)
        roi = batch.arrays[self.raw].spec.roi
        log[0] = np.copy(roi.get_offset())
        log[1] = np.copy(roi.get_shape())

        logger.debug(f'Block offset and shape: {log}')

        batch = gp.Batch()
        batch[self.log] = gp.Array(
            log, gp.ArraySpec(nonspatial=True)
        )

        return batch
