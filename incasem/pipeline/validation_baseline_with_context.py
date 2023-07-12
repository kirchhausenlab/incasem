import logging
import os
import numpy as np

import gunpowder as gp
import incasem as fos

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ValidationBaselineWithContext:
    def __init__(
            self,
            data_config,
            run_dir,
            run_path_prefix,
            data_path_prefix,
            model,
            loss,
            num_classes,
            voxel_size,
            input_size_voxels,
            output_size_voxels,
            run_every,
            random_seed=None,
    ):
        self._data_config = data_config[0]
        self._data_config_name = data_config[1]

        self._run_dir = run_dir
        self._run_path_prefix = run_path_prefix
        self._data_path_prefix = data_path_prefix
        self._model = model
        self._loss = loss
        self._num_classes = num_classes

        self._voxel_size = gp.Coordinate(voxel_size)
        self._input_size = self._voxel_size * gp.Coordinate(input_size_voxels)
        self._output_size = self._voxel_size * \
            gp.Coordinate(output_size_voxels)

        self._run_every = run_every
        self._random_seed = random_seed

        self._assemble_pipeline()

    def _assemble_pipeline(self):
        # define pipeline
        keys = {
            'RAW': gp.ArrayKey('RAW'),
            'LABELS': gp.ArrayKey('LABELS'),
            'MASK': gp.ArrayKey('MASK'),
            'METRIC_MASK': gp.ArrayKey('METRIC_MASK'),
            'LOSS_SCALINGS': gp.ArrayKey('LOSS_SCALINGS'),
            'PREDICTIONS': gp.ArrayKey('PREDICTIONS'),
        }

        self.request = gp.BatchRequest(random_seed=self._random_seed)
        self.request.add(keys['RAW'], self._input_size)
        self.request.add(keys['LABELS'], self._output_size)
        self.request.add(keys['MASK'], self._output_size)
        self.request.add(keys['METRIC_MASK'], self._output_size)
        self.request.add(keys['LOSS_SCALINGS'], self._output_size)
        self.request.add(keys['PREDICTIONS'], self._output_size)

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

        if len(sources.pipelines) > 1:
            raise NotImplementedError(
                "Validation only implemented for a single input dataset.")
        self.pipeline = sources.pipelines[0]

        self.downsample = fos.gunpowder.Downsample(
            source=[
                keys['RAW'],
                keys['LABELS'],
                keys['MASK'],
                keys['METRIC_MASK'],
            ],
            target=[
                keys['RAW'],
                keys['LABELS'],
                keys['MASK'],
                keys['METRIC_MASK'],
            ],
            factor=1,
        )
        self.pipeline = (
            self.pipeline
            + self.downsample
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
            + fos.gunpowder.ToDtype([keys['LABELS']], dtype='int64')
            + gp.IntensityScaleShift(keys['RAW'], 2, -1)

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

        # The predict node needs to be told which ROI of predictions it
        # provides.
        validation_roi = sources.rois[0].copy()
        # If validation ROI is smaller than the reference request, grow it to
        # that size.
        validation_roi = fos.utils.grow_roi_to(
            roi=validation_roi,
            target_shape=self._output_size,
            voxel_size=self._voxel_size
        )
        logger.debug(f"{validation_roi=}")

        self.predict = fos.gunpowder.torch.Predict(
            model=self._model,
            inputs={
                'x': keys['RAW'],
            },
            outputs={0: keys['PREDICTIONS']},
            array_specs={
                keys['PREDICTIONS']: gp.ArraySpec(
                    roi=validation_roi,
                    dtype=np.float32,
                    voxel_size=self._voxel_size
                )
            },
        )
        self.pipeline = (
            self.pipeline
            + self.predict
        )

        self.scan = gp.Scan(reference=self.request, num_workers=1)
        self.pipeline = (
            self.pipeline +
            self.scan
        )

        self.validation_loss = fos.gunpowder.torch.ValidationLoss(
            loss=self._loss,
            inputs={
                0: keys['PREDICTIONS'],
                1: keys['LABELS'],
                2: keys['MASK'],
                3: keys['LOSS_SCALINGS'],
            },
            log_dir=os.path.join(
                self._run_path_prefix,
                'tensorboard',
                self._run_dir,
                'validation',
                self._data_config_name
            ),
            log_every=self._run_every
        )
        self.pipeline = (
            self.pipeline
            + self.validation_loss
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

        self.snapshot = fos.gunpowder.Snapshot(
            dataset_names={
                v: f"{k.lower()}" for k, v in keys.items()
            },
            every=1,
            output_dir=os.path.join(
                self._run_path_prefix,
                'snapshots',
                self._run_dir,
                'validation',
                self._data_config_name,
            ),
            output_filename='{iteration}.zarr',
            compression_type='zlib',
            compression_level=3,
            chunk_shape=(128, 128, 128),
        )
        self.pipeline = (
            self.pipeline
            + self.snapshot
            + fos.gunpowder.Uint8ToFloat(keys['PREDICTIONS'])
            + gp.PrintProfilingStats(every=1)
        )
