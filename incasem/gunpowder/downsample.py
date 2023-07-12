import logging
import numbers

from gunpowder.nodes.batch_filter import BatchFilter
from gunpowder.array import ArrayKey, Array
from gunpowder.batch_request import BatchRequest
from gunpowder.batch import Batch

logger = logging.getLogger(__name__)


class Downsample(BatchFilter):
    """Downsample arrays in a batch by given factors.

    Args:

        source (:class:`ArrayKey` or ``list`` of :class:`ArrayKey`):

            The key(s) of the array(s) to downsample.

        factor (``int`` or ``tuple`` of ``int``):

            The factors(s) to downsample with.

        target (:class:`ArrayKey` or ``list`` of :class:`ArrayKey`):

            The keys(s) of the array(s) to store the downsampled `source`(s).
            The number of `target`s has to match the number of `source`s.
    """

    def __init__(self, source, factor, target):

        if isinstance(source, ArrayKey):
            self.source = [source]
        else:
            self.source = source
        for s in self.source:
            assert isinstance(s, ArrayKey)

        assert isinstance(factor, (numbers.Number, tuple)), \
            "Scaling factor should be a number or a tuple of numbers."
        self.factor = factor

        if isinstance(target, ArrayKey):
            self.target = [target]
        else:
            self.target = target
        for t in self.target:
            assert isinstance(t, ArrayKey)

        if len(self.source) != len(self.target):
            raise ValueError(
                "Number of sources and target arrays does not match.")

    def setup(self):

        self.enable_autoskip()

        for source, target in zip(self.source, self.target):
            spec = self.spec[source].copy()
            spec.voxel_size *= self.factor

            if source == target:
                self.updates(source, spec)
            else:
                self.provides(target, spec)

    def prepare(self, request):

        deps = BatchRequest()

        for source, target in zip(self.source, self.target):
            deps[source] = request[target].copy()

        return deps

    def process(self, batch, request):

        outputs = Batch()

        for source, target in zip(self.source, self.target):
            # downsample
            if isinstance(self.factor, tuple):
                slices = tuple(
                    slice(None, None, k)
                    for k in self.factor)
            else:
                slices = tuple(
                    slice(None, None, self.factor)
                    for i in range(batch[source].spec.roi.dims()))

            logger.debug(f"downsampling {source} with {slices}")

            data = batch.arrays[source].data[slices]

            # create output array
            spec = self.spec[target].copy()
            spec.roi = request[target].roi
            outputs.arrays[target] = Array(data, spec)

        return outputs
