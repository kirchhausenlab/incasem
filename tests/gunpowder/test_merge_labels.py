import copy
import pytest
import numpy as np
import gunpowder as gp

import incasem as fos


class Source(gp.BatchProvider):
    def __init__(self, voxel_size, key_a="A", key_b="B"):
        self.voxel_size = gp.Coordinate(voxel_size)
        self.roi = gp.Roi((0, 0, 0), (10, 10, 10)) * self.voxel_size

        self.raw = gp.ArrayKey("RAW")
        self.a = gp.ArrayKey(key_a)
        self.b = gp.ArrayKey(key_b)

        self.array_spec_raw = gp.ArraySpec(
            roi=self.roi,
            voxel_size=self.voxel_size,
            dtype='uint8',
            interpolatable=True
        )

        self.array_spec_labels = gp.ArraySpec(
            roi=self.roi,
            voxel_size=self.voxel_size,
            dtype='uint8',
            interpolatable=False
        )

    def setup(self):

        self.provides(self.raw, self.array_spec_raw)
        self.provides(self.a, self.array_spec_labels)
        self.provides(self.b, self.array_spec_labels)

    def provide(self, request):
        outputs = gp.Batch()

        # RAW
        raw_spec = copy.deepcopy(self.array_spec_raw)
        raw_spec.roi = request[self.raw].roi

        outputs[self.raw] = gp.Array(
            np.random.randint(
                0,
                256,
                request[self.raw].roi.get_shape() / self.voxel_size,
                dtype=raw_spec.dtype
            ),
            raw_spec
        )

        # A
        a_spec = copy.deepcopy(self.array_spec_labels)
        a_spec.roi = request[self.a].roi
        a = np.zeros(
            request[self.a].roi.get_shape() / self.voxel_size,
            dtype=a_spec.dtype
        )
        a[0, :, :] = 1
        a[1, 0, 0] = 1
        outputs[self.a] = gp.Array(a, a_spec)

        # B
        b_spec = copy.deepcopy(self.array_spec_labels)
        b_spec.roi = request[self.b].roi
        b = np.zeros(
            request[self.b].roi.get_shape() / self.voxel_size,
            dtype=b_spec.dtype
        )
        b[1, :, :] = 1
        outputs[self.b] = gp.Array(b, b_spec)

        return outputs


@pytest.mark.parametrize("voxel_size", [(1, 1, 1), (5, 5, 5)])
@pytest.mark.parametrize("ambiguous_labels", ['max', 'background'])
def test_merge_labels(voxel_size, ambiguous_labels):
    raw = gp.ArrayKey("RAW")
    a = gp.ArrayKey("A")
    b = gp.ArrayKey("B")
    labels = gp.ArrayKey("LABELS")

    voxel_size = gp.Coordinate(voxel_size)
    input_voxels = gp.Coordinate((10, 10, 10))
    input_size = input_voxels * voxel_size

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, input_size)

    pipeline = (
        Source(voxel_size)
        + fos.gunpowder.MergeLabels(
            classes={a: 1, b: 2},
            output_array=labels,
            ambiguous_labels=ambiguous_labels
        )
        + gp.Normalize(raw)
    )

    expected_output = np.zeros((10, 10, 10), dtype=np.uint64)
    expected_output[0, :, :] = 1
    expected_output[1, :, :] = 2

    if ambiguous_labels == 'max':
        expected_output[1, 0, 0] = 2
    elif ambiguous_labels == 'background':
        expected_output[1, 0, 0] = 0
    else:
        raise ValueError()

    with gp.build(pipeline) as p:
        batch = p.request_batch(request)
        assert batch[labels].data.shape == batch[raw].data.shape
        assert batch[labels].data.shape == input_voxels
        assert (batch[labels].data == expected_output).all()


@pytest.mark.parametrize("voxel_size", [(1, 1, 1), (5, 5, 5)])
def test_merge_labels_and_augment(voxel_size):
    raw = gp.ArrayKey("RAW")
    a = gp.ArrayKey("A")
    b = gp.ArrayKey("B")
    labels = gp.ArrayKey("LABELS")

    voxel_size = gp.Coordinate(voxel_size)
    input_voxels = gp.Coordinate((10, 10, 10))
    input_size = input_voxels * voxel_size

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, input_size)

    pipeline = (
        Source(voxel_size)
        + fos.gunpowder.MergeLabels(
            classes={a: 1, b: 2},
            output_array=labels
        )
        + gp.Normalize(raw)
        + gp.RandomLocation()
        # Padding to avoid requests bigger than the total roi
        + fos.gunpowder.PadDownstreamOfRandomLocation(raw, None)
        + fos.gunpowder.PadDownstreamOfRandomLocation(labels, None)
        + gp.ElasticAugment(
            control_point_spacing=(5, 5, 5),
            jitter_sigma=(2, 2, 2),
            rotation_interval=(0, np.pi / 2)
        )
    )

    expected_output = np.zeros((10, 10, 10), dtype=np.uint64)
    expected_output[0, :, :] = 1
    expected_output[1, :, :] = 2

    with gp.build(pipeline) as p:
        batch = p.request_batch(request)
        assert batch[labels].data.shape == batch[raw].data.shape
        assert batch[labels].data.shape == input_voxels
        # assert batch[labels].data.sum() == 100 * 1 + 100 * 2
        # assert (batch[labels].data == expected_output).all()


@pytest.mark.parametrize("voxel_size", [(1, 1, 1), (5, 5, 5)])
def test_merge_labels_and_random_provider(voxel_size):
    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")

    voxel_size = gp.Coordinate(voxel_size)
    input_voxels = gp.Coordinate((10, 10, 10))
    input_size = input_voxels * voxel_size

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, input_size)

    pipelines = []
    for i in range(2):
        a_string = f"A_{i}"
        b_string = f"B_{i}"
        a = gp.ArrayKey(a_string)
        b = gp.ArrayKey(b_string)
        pipeline = (
            Source(voxel_size, key_a=a_string, key_b=b_string)
            + fos.gunpowder.BinarizeLabels([a, b])
            + fos.gunpowder.MergeLabels(
                classes={a: 1, b: 2},
                output_array=labels
            )
            + gp.Normalize(raw)
            + gp.RandomLocation()
            # Padding to avoid requests bigger than the total roi
            # + fos.gunpowder.PadDownstreamOfRandomLocation(raw, None)
            # + fos.gunpowder.PadDownstreamOfRandomLocation(labels, None)
            # + gp.ElasticAugment(
            # control_point_spacing=(5, 5, 5),
            # jitter_sigma=(2, 2, 2),
            # rotation_interval=(0, np.pi / 2)
            # )
        )
        pipelines.append(pipeline)

    pipeline = (
        tuple(pipelines)
        + gp.RandomProvider()
    )

    with gp.build(pipeline) as p:
        batch = p.request_batch(request)
        assert batch[labels].data.shape == batch[raw].data.shape
        assert batch[labels].data.shape == input_voxels
        # assert batch[labels].data.sum() == 100 * 1 + 100 * 2
        # assert (batch[labels].data == expected_output).all()


if __name__ == '__main__':
    test_merge_labels_and_random_provider((5, 5, 5))
