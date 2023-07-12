import logging
import numpy as np

from gunpowder.nodes.batch_filter import BatchFilter
from gunpowder.array import ArrayKey
from gunpowder.coordinate import Coordinate
from gunpowder.roi import Roi
from gunpowder.batch_request import BatchRequest
from gunpowder import ArraySpec

logger = logging.getLogger(__name__)


class PadDownstreamOfRandomLocation(BatchFilter):
    '''Add a constant intensity padding around arrays of another batch
    provider. This is useful if your requested batches can be larger than what
    your source provides.

    Args:

        key (:class:`ArrayKey` or :class:`GraphKey`):

            The array or points set to pad.

        size (:class:`Coordinate` or ``None``):

            The padding to be added. If None, an infinite padding is added. If
            a coordinate, this amount will be added to the ROI in the positive
            and negative direction.

        value (scalar or ``None``):

            The value to report inside the padding. If not given, 0 is used.
            Only used for :class:`Array<Arrays>`.
    '''

    def __init__(self, key, size, value=None):

        self.key = key
        self.size = size
        self.value = value

    def setup(self):
        self.enable_autoskip()

        assert self.key in self.spec, (
            "Asked to pad %s, but is not provided upstream." % self.key)
        assert self.spec[self.key].roi is not None, (
            "Asked to pad %s, but upstream provider doesn't have a ROI for "
            "it." % self.key)

        spec = self.spec[self.key].copy()
        if self.size is not None:
            spec.roi = spec.roi.grow(self.size, self.size)
        else:
            spec.roi.set_shape(None)
        self.updates(self.key, spec)

    def prepare(self, request):

        # go upstream to a provider that does not provide a infinite array
        rec_upstream_provider = self.get_upstream_provider()
        while rec_upstream_provider.spec[self.key].roi.get_shape(
        ) == Coordinate((None, None, None)):
            rec_upstream_provider = \
                rec_upstream_provider.get_upstream_provider()

        upstream_spec = rec_upstream_provider.spec

        assert isinstance(
            upstream_spec[self.key], ArraySpec), \
            f'{__name__} only implemented for Arrays'

        logger.debug("request: %s" % request)
        logger.debug("upstream spec: %s" % upstream_spec)

        new_shape = Coordinate([u if u < r else r for (u, r) in
                                zip(upstream_spec[self.key].roi.get_shape(),
                                    request[self.key].roi.get_shape())
                                ])

        voxel_size = upstream_spec[self.key].voxel_size
        shrink_coordinate = (
            (request[self.key].roi.get_shape() - new_shape + voxel_size)
            // voxel_size
        ) // 2
        shrink_coordinate = shrink_coordinate * voxel_size

        request[self.key].roi = request[self.key].roi.grow(
            -shrink_coordinate,
            -shrink_coordinate
        )

        logger.debug("new request: %s" % request)

        deps = BatchRequest()
        deps[self.key] = request[self.key]
        return deps

    def process(self, batch, request):

        if self.key not in request:
            return

        # restore requested batch size and ROI
        if isinstance(self.key, ArrayKey):

            array = batch.arrays[self.key]
            array.data = self.__expand(
                array.data,
                array.spec.roi / array.spec.voxel_size,
                request[self.key].roi / array.spec.voxel_size,
                self.value if self.value else 0
            )
            array.spec.roi = request[self.key].roi

        else:

            points = batch.points[self.key]
            points.spec.roi = request[self.key].roi

    def __expand(self, a, from_roi, to_roi, value):
        '''from_roi and to_roi should be in voxels.'''

        logger.debug(
            "expanding array of shape %s from %s to %s",
            str(a.shape), from_roi, to_roi)

        num_channels = len(a.shape) - from_roi.dims()
        channel_shapes = a.shape[:num_channels]

        b = np.zeros(channel_shapes + to_roi.get_shape(), dtype=a.dtype)
        if value != 0:
            b[:] = value

        shift = tuple(-x for x in to_roi.get_offset())
        logger.debug("shifting 'from' by " + str(shift))
        a_in_b = from_roi.shift(shift).to_slices()

        logger.debug("target shape is " + str(b.shape))
        logger.debug("target slice is " + str(a_in_b))

        b[(slice(None),) * num_channels + a_in_b] = a

        return b
