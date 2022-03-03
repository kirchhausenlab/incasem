import logging
import numpy as np

from gunpowder.nodes.batch_filter import BatchFilter
from gunpowder.array import ArrayKey
from gunpowder.coordinate import Coordinate
from gunpowder.points import PointsKey
from gunpowder.roi import Roi
from gunpowder.batch_request import BatchRequest

logger = logging.getLogger(__name__)


class PadTo(BatchFilter):
    '''Add a constant intensity padding around arrays of another batch
    provider. This is useful if your requested batches can be larger than what
    your source provides.

    Args:

        key (:class:`ArrayKey` or :class:`GraphKey`):

            The array or points set to pad.

        # TODO adapt docstring
        size (:class:`Coordinate` or ``None``):

            The padding to be added. If None, an infinite padding is added. If
            a coordinate, this amount will be added to the ROI in the positive
            and negative direction.

        value (scalar or ``None``):

            The value to report inside the padding. If not given, 0 is used.
            Only used for :class:`Array<Arrays>`.
    '''

    def __init__(self, key, target_size, value=None):

        self.key = key
        self.target_size = target_size
        self.value = value

    def setup(self):
        self.enable_autoskip()

        assert self.key in self.spec, (
            "Asked to pad %s, but is not provided upstream." % self.key)
        assert self.spec[self.key].roi is not None, (
            "Asked to pad %s, but upstream provider doesn't have a ROI for "
            "it." % self.key)

        assert self.target_size is not None, (
            f"No target_size provided."
        )

        spec = self.spec[self.key].copy()

        grow_size_total = self.target_size - spec.roi.get_shape()
        # only grow if the spec roi is smaller than the target roi
        grow_size_total = Coordinate(
            (max(g, 0) for g in grow_size_total)
        )

        grow_size_neg = (grow_size_total // spec.voxel_size) // 2
        grow_size_pos = ((grow_size_total + spec.voxel_size)
                         // spec.voxel_size) // 2
        spec.roi = spec.roi.grow(
            grow_size_neg * spec.voxel_size,
            grow_size_pos * spec.voxel_size
        )

        self.updates(self.key, spec)

    def prepare(self, request):

        upstream_spec = self.get_upstream_provider().spec

        logger.debug("request: %s" % request)
        logger.debug("upstream spec: %s" % upstream_spec)

        # TODO: remove this?
        if self.key not in request:
            return

        roi = request[self.key].roi.copy()

        # change request to fit into upstream spec
        request[self.key].roi = roi.intersect(upstream_spec[self.key].roi)

        if request[self.key].roi.empty():

            logger.warning(
                "Requested %s ROI %s lies entirely outside of upstream "
                "ROI %s.", self.key, roi, upstream_spec[self.key].roi)

            # ensure a valid request by asking for empty ROI
            request[self.key].roi = Roi(
                upstream_spec[self.key].roi.get_offset(),
                (0,) * upstream_spec[self.key].roi.dims()
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
