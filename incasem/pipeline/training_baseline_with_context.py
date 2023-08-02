import os
import logging
import numpy as np

import gunpowder as gp
import incasem as fos

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TrainingBaselineWithContext:
    """Training pipeline with U-Net for semantic segmentation.

    Dataset has to fit in memory.
    """

    def __init__(
            self,
            data_config,
            run_dir,
            run_path_prefix,
            data_path_prefix,
            model,
            loss,
            optimizer,
            num_classes,
            voxel_size,
            input_size_voxels,
            output_size_voxels,
            reject_min_masked=0.05,
            reject_probability=0.9,
            random_seed=None,
    ):
        self._data_config = data_config
        self._run_dir = run_dir
        self._run_path_prefix = run_path_prefix
        self._data_path_prefix = data_path_prefix
        self._model = model
        self._loss = loss
        self._optimizer = optimizer
        self._num_classes = num_classes

        self._voxel_size = gp.Coordinate(voxel_size)
        self._input_size = self._voxel_size * gp.Coordinate(input_size_voxels)
        self._output_size = self._voxel_size * \
            gp.Coordinate(output_size_voxels)

        self._reject_min_masked = reject_min_masked
        self._reject_probability = reject_probability
        self._random_seed = random_seed

        self._assemble_pipeline()

    def _assemble_pipeline(self):

        keys = {
            'RAW': gp.ArrayKey('RAW'),
            'RAW_OUTPUT_SIZE': gp.ArrayKey('RAW_OUTPUT_SIZE'),
            'LABELS': gp.ArrayKey('LABELS'),
            'MASK': gp.ArrayKey('MASK'),
            'BACKGROUND_MASK': gp.ArrayKey('BACKGROUND_MASK'),
            'METRIC_MASK': gp.ArrayKey('METRIC_MASK'),
            'LOSS_SCALINGS': gp.ArrayKey('LOSS_SCALINGS'),
            'PREDICTIONS': gp.ArrayKey('PREDICTIONS'),
        }

        raw_pos = gp.ArrayKey('RAW_POS')

        self.request = gp.BatchRequest(random_seed=self._random_seed)
        self.request.add(keys['RAW'], self._input_size)
        self.request.add(keys['RAW_OUTPUT_SIZE'], self._output_size)
        self.request.add(keys['LABELS'], self._output_size)
        self.request.add(keys['MASK'], self._output_size)
        self.request.add(keys['BACKGROUND_MASK'], self._output_size)
        self.request.add(keys['METRIC_MASK'], self._output_size)
        self.request.add(keys['LOSS_SCALINGS'], self._output_size)
        self.request.add(keys['PREDICTIONS'], self._output_size)

        self.request[raw_pos] = gp.ArraySpec(nonspatial=True)

        context = (self._input_size -
                   self._output_size) // self._voxel_size // 2
        sources = fos.pipeline.sources.DataSourcesSemanticWithContext(
            config_file=self._data_config,
            keys=keys,
            data_path_prefix=self._data_path_prefix,
            context=context,
        )

        if self._voxel_size != sources.voxel_size:
            logger.warning((
                f"Specfied voxel size {self._voxel_size} does not match "
                f"the voxel size of the datasets {sources.voxel_size}."
            ))

        pipelines_with_random_locations = []
        for sources_p in sources.pipelines:
            p = (
                sources_p
                + fos.gunpowder.DeepCopyArrays(
                    arrays=[keys['LABELS']],
                    output_arrays=[keys['BACKGROUND_MASK']]
                )
                + fos.gunpowder.BinarizeLabels([keys['BACKGROUND_MASK']])

                + fos.gunpowder.SaveBlockPosition(
                    keys['RAW'],
                    raw_pos
                )
                + fos.gunpowder.RandomLocationBounded(
                    mask=keys['BACKGROUND_MASK'],
                    min_masked=self._reject_min_masked,
                    reject_probability=self._reject_probability,
                )

                + fos.gunpowder.PadDownstreamOfRandomLocation(
                    keys['RAW'],
                    size=None,
                    value=0.5
                )
                + fos.gunpowder.PadDownstreamOfRandomLocation(
                    keys['LABELS'],
                    size=None,
                    value=0
                )
                # Padding mask with 1s to make sure that shallow blocks are
                # used for training
                + fos.gunpowder.PadDownstreamOfRandomLocation(
                    keys['MASK'],
                    size=None,
                    value=1
                )
                + fos.gunpowder.PadDownstreamOfRandomLocation(
                    keys['BACKGROUND_MASK'],
                    size=None,
                    value=0
                )
                + fos.gunpowder.PadDownstreamOfRandomLocation(
                    keys['METRIC_MASK'],
                    size=None,
                    value=1
                )

                + fos.gunpowder.CentralizeRequests()
            )
            pipelines_with_random_locations.append(p)

        # TODO if working with prelodaded integral array for sampling,
        # the probs should ideally be proportional to masked in voxels
        probabilities = [r.size() for r in sources.rois]
        probs_dict = {
            name: float(prob) / np.sum(probabilities)
            for name, prob in zip(sources.names, probabilities)
        }
        logger.info(
            f"Sampling probabilities for the provided datasets:\n"
            f"{probs_dict}"
        )

        self.pipeline = (
            tuple(pipelines_with_random_locations)
            + gp.RandomProvider(probabilities)
        )

        # Reject blocks that contain non-sample
        self.pipeline = (
            self.pipeline
            + fos.gunpowder.Reject(
                mask=keys['MASK'],
                min_masked=0.75,
                reject_probability=0.95,
            )
        )

        self.downsample = fos.gunpowder.Downsample(
            source=[
                keys['RAW'],
                keys['LABELS'],
                keys['MASK'],
                keys['BACKGROUND_MASK'],
                keys['METRIC_MASK'],
            ],
            target=[
                keys['RAW'],
                keys['LABELS'],
                keys['MASK'],
                keys['BACKGROUND_MASK'],
                keys['METRIC_MASK'],
            ],
            factor=1,
        )
        self.pipeline = (
            self.pipeline
            + self.downsample
        )

        # Augmentation
        self.augmentation = fos.pipeline.sections.Augmentation(
            raw_key=keys['RAW'],
        )

        # TODO these calls should be ported to the Section base class as a
        # single call to add a section to a pipeline

        self.pipeline = self.augmentation.add_to_pipeline(self.pipeline)
        keys = {**keys, **self.augmentation.new_keys}
        self.request.merge(self.augmentation.new_request)

        self.pipeline = (
            self.pipeline
            + fos.gunpowder.ToDtype([keys['LABELS']], dtype='int64')
        )

        self.balance_labels = gp.BalanceLabels(
            labels=keys['LABELS'],
            scales=keys['LOSS_SCALINGS'],
            num_classes=self._num_classes,
            mask=keys['MASK']
        )
        self.pipeline = (
            self.pipeline
            + self.balance_labels
        )

        # Prepare data format for model
        self.pipeline = (
            self.pipeline
            + gp.IntensityScaleShift(keys['RAW'], 2, -1)

            # Some Augmentation nodes lead to negative numpy strides,
            # which breaks typecasting to torch tensor
            + fos.gunpowder.DeepCopy(
                [keys['RAW'], keys['LABELS'], keys['MASK']]
            )

            # Create channel dimension, but only for the raw input
            + fos.gunpowder.Unsqueeze([keys['RAW']])
            # Create batch dimension
            + fos.gunpowder.Unsqueeze([
                keys['RAW'],
                keys['LABELS'],
                keys['MASK'],
                keys['LOSS_SCALINGS'],
            ])
        )

        self.precache = gp.PreCache(cache_size=10, num_workers=5)
        self.pipeline = (
            self.pipeline
            + self.precache
        )

        # checkpoints in some directory
        checkpoint_dir = os.path.join(
            self._run_path_prefix,
            "models",
            self._run_dir
        )
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        loss_inputs = {
            0: keys['PREDICTIONS'],
            1: keys['LABELS'],
            2: keys['MASK'],
            3: keys['LOSS_SCALINGS'],
        }

        self.train_node = fos.gunpowder.torch.Train(
            model=self._model,
            loss=self._loss,
            optimizer=self._optimizer,
            inputs={
                'x': keys['RAW'],
            },
            outputs={0: keys['PREDICTIONS']},
            loss_inputs=loss_inputs,
            array_specs={
                keys['PREDICTIONS']: gp.ArraySpec(
                    dtype=np.float32, voxel_size=self._voxel_size
                )
            },
            checkpoint_basename=os.path.join(checkpoint_dir, 'model'),
            save_every=500,
            log_dir=os.path.join(
                self._run_path_prefix,
                'tensorboard',
                self._run_dir,
                'training'
            ),
            log_every=1,
        )
        self.pipeline = (
            self.pipeline
            + self.train_node
        )

        self.pipeline = (
            self.pipeline
            + fos.gunpowder.Squeeze([
                keys['RAW'],
                keys['LABELS'],
                keys['MASK'],
                keys['LOSS_SCALINGS'],
                keys['PREDICTIONS'],
            ])
            + fos.gunpowder.Squeeze([keys['RAW']])
        )

        self.pipeline = (
            self.pipeline
            # scale back intensity to [0,1] interval
            + gp.IntensityScaleShift(keys['RAW'], 0.5, 0.5)
            + fos.gunpowder.FloatToUint8(keys['RAW'])
            + fos.gunpowder.ToDtype([keys['LABELS']], dtype='uint32')
        )

        self.pipeline = (
            self.pipeline
            + fos.gunpowder.Softmax(keys['PREDICTIONS'])
            + fos.gunpowder.FloatToUint8(keys['PREDICTIONS'])
        )

        self.pipeline = (
            self.pipeline
            + fos.gunpowder.DeepCopyArrays(
                arrays=[keys['RAW']],
                output_arrays=[keys['RAW_OUTPUT_SIZE']]
            )
        )
        self.snapshot = fos.gunpowder.Snapshot(
            dataset_names={
                v: f"{k.lower()}" for k, v in keys.items()
            },
            every=1,
            output_dir=os.path.join(
                self._run_path_prefix,
                'snapshots',
                self._run_dir,
                'training'
            ),
            output_filename='{iteration}.zarr',
            compression_type='zlib',
            compression_level=3,
            chunk_shape=tuple(self._output_size / self._voxel_size),
        )

        self.profiling_stats = gp.PrintProfilingStats(every=10)
        self.pipeline = (
            self.pipeline
            + self.snapshot
            + fos.gunpowder.Uint8ToFloat(keys['PREDICTIONS'])
            + self.profiling_stats
        )

        logger.debug(self.pipeline)
