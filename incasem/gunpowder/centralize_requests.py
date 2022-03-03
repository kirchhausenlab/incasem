import logging

from gunpowder.nodes.batch_filter import BatchFilter
from gunpowder.coordinate import Coordinate
from gunpowder.batch_request import BatchRequest

logger = logging.getLogger(__name__)


class CentralizeRequests(BatchFilter):
    """
    """

    def __init__(self):
        self.grow_amounts = {}

    # def setup(self):
        # self.enable_autoskip()
        # self.updates(self.key, spec)

    def prepare(self, request):

        # find biggest ROI in request
        big_key = None

        for key, spec in request.items():
            if spec.nonspatial:
                continue

            if big_key is None:
                big_key = key

            if all((s > b for s, b in zip(spec.roi.get_shape(),
                                          request[big_key].roi.get_shape()))):
                big_key = key

        logger.debug(f"{big_key=}")

        # for each array, calculate distance to biggest ROI from its beginning
        # and end
        for key, spec in request.items():
            if spec.nonspatial:
                continue

            logger.debug(f"{key=}")
            left_indent = spec.roi.get_begin() - \
                request[big_key].roi.get_begin()
            right_indent = request[big_key].roi.get_end() - \
                spec.roi.get_end()

            difference = Coordinate(abs(l - r)
                                    for l, r in zip(left_indent, right_indent))
            logger.debug(f"{difference=}")

            roi = request[key].roi.copy()
            grow_left = [0] * self.spec[key].voxel_size.dims()
            grow_right = [0] * self.spec[key].voxel_size.dims()

            for i, (l, r) in enumerate(zip(left_indent, right_indent)):
                if l > r:
                    grow_left[i] = difference[i]

                if l < r:
                    grow_right[i] = difference[i]

            logger.debug(f"{grow_left=}")
            logger.debug(f"{grow_right=}")

            grow_left = Coordinate(grow_left)
            grow_right = Coordinate(grow_right)

            roi = roi.grow(
                grow_left,
                grow_right
            )

            request[key].roi = roi

            self.grow_amounts[key] = (grow_left, grow_right)

        logger.debug("new request: %s" % request)

        deps = BatchRequest()
        for key, spec in request.items():
            if spec.nonspatial:
                continue

            deps[key] = request[key].copy()

        return deps

    def process(self, batch, request):

        # shrink data according to the stored growth Coordinates
        for key, spec in request.items():
            if spec.nonspatial:
                continue

            logger.debug(f"Process {key}")
            array = batch.arrays[key]
            target_roi = array.spec.roi

            left_grow, right_grow = self.grow_amounts[key]
            target_roi.grow(
                -left_grow,
                -right_grow
            )

            array.data = self.__shrink(
                array.data,
                array.spec.roi / array.spec.voxel_size,
                request[key].roi / array.spec.voxel_size
            )

            array.spec.roi = request[key].roi

    def __shrink(self, a, from_roi, to_roi):
        '''from_roi and to_roi should be in voxels.'''

        logger.debug(
            f"shrinking array of shape {a.shape} from {from_roi} to {to_roi}")

        num_channels = len(a.shape) - from_roi.dims()

        # shift to 0
        shift = Coordinate((-x for x in from_roi.get_offset()))
        logger.debug("shifting both ROIs by " + str(shift))

        from_roi = from_roi + shift
        to_roi = to_roi + shift

        logger.debug(f"{from_roi=}")
        logger.debug(f"{to_roi=}")

        to_in_from = to_roi.to_slices()

        logger.debug("target shape is " + str(to_roi.get_shape()))
        logger.debug("target slice is " + str(to_in_from))

        out = a[(slice(None),) * num_channels + to_in_from]

        logger.debug(f"{out.shape=}")

        return out
