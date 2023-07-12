import logging

import copy
import pytest
import numpy as np
import gunpowder as gp

import incasem as fos

logging.basicConfig(level=logging.INFO)
logging.getLogger('gunpowder').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Source(gp.BatchProvider):
    def __init__(self, raw_key, voxel_size=(1, 1, 1)):
        self.voxel_size = gp.Coordinate(voxel_size)
        self.roi = gp.Roi((0, 0, 0), (20, 20, 20)) * self.voxel_size

        self.raw_key = raw_key
        self.array_spec_raw = gp.ArraySpec(
            roi=self.roi,
            voxel_size=self.voxel_size,
            dtype='uint8',
            interpolatable=True
        )

    def setup(self):
        self.provides(self.raw_key, self.array_spec_raw)

    def provide(self, request):
        outputs = gp.Batch()

        # create array spec
        array_spec = copy.deepcopy(self.array_spec_raw)
        array_spec.roi = request[self.raw_key].roi

        outputs[self.raw_key] = gp.Array(
            np.random.randint(
                0,
                256,
                request[self.raw_key].roi.get_shape() / self.voxel_size,
                dtype=array_spec.dtype
            ),
            array_spec
        )

        return outputs


@pytest.mark.parametrize("value", [0, 5])
@pytest.mark.parametrize("voxel_size", [(1, 1, 1), (5, 5, 5)])
def test_add_background_labels(value, voxel_size):
    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")

    voxel_size = gp.Coordinate(voxel_size)
    input_size = gp.Coordinate((10, 10, 10)) * voxel_size

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, input_size)

    pipeline = (
        Source(raw, voxel_size)
        + fos.gunpowder.AddBackgroundLabels(
            raw_array=raw,
            output_array=labels,
            value=value
        )
    )

    expected_output = np.full((10, 10, 10), value, dtype=np.uint64)

    with gp.build(pipeline) as p:
        batch = p.request_batch(request)
        assert batch[labels].data.shape == batch[raw].data.shape
        assert batch[labels].data.shape == (10, 10, 10)
        assert batch[labels].data.sum() == 10 * 10 * 10 * value
        assert (batch[labels].data == expected_output).all()


@pytest.mark.parametrize("value", [0, 3, 5])
@pytest.mark.parametrize("voxel_size", [(1, 1, 1), (5, 5, 5)])
def test_add_background_labels_and_augment(value, voxel_size):
    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")

    input_voxels = gp.Coordinate((10, 10, 10))
    input_size = input_voxels * gp.Coordinate(voxel_size)

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, input_size)

    pipeline = (
        Source(raw, gp.Coordinate(voxel_size))
        + fos.gunpowder.AddBackgroundLabels(
            raw_array=raw,
            output_array=labels,
            value=value
        )
        + gp.Normalize(raw)
        + gp.RandomLocation()
        # Padding to avoid requests bigger than the total roi
        + fos.gunpowder.PadDownstreamOfRandomLocation(raw, None)
        + fos.gunpowder.PadDownstreamOfRandomLocation(labels, None, value)
        + gp.ElasticAugment(
            control_point_spacing=(5, 5, 5),
            jitter_sigma=(2, 2, 2),
            rotation_interval=(0, np.pi / 2)
        )
    )

    expected_output = np.full(input_voxels, value, dtype=np.uint64)

    with gp.build(pipeline) as p:
        batch = p.request_batch(request)
        assert batch[labels].data.shape == batch[raw].data.shape
        assert batch[labels].data.shape == input_voxels
        assert batch[labels].data.sum() == 10 * 10 * 10 * value
        assert (batch[labels].data == expected_output).all()


if __name__ == '__main__':
    test_add_background_labels_and_augment(0, (5, 5, 5))
