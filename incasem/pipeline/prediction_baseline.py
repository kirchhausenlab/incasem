import logging
import os

import numpy as np

import gunpowder as gp
import incasem as fos

logger = logging.getLogger(__name__)


class PredictionBaseline:
    """Prediction pipeline with U-Net for semantic segmentation.
    """

    def __init__(
            self,
            data_config,
            run_id,
            data_path_prefix,
            predictions_path_prefix,
            model,
            num_classes,
            voxel_size,
            input_size_voxels,
            output_size_voxels,
            checkpoint,
    ):
        self._data_config = data_config[0]
        self._data_config_name = data_config[1]

        self._run_id = run_id
        self._data_path_prefix = data_path_prefix
        self._predictions_path_prefix = predictions_path_prefix
        self._model = model
        self._num_classes = num_classes

        self._voxel_size = gp.Coordinate(voxel_size)
        self._input_size = self._voxel_size * gp.Coordinate(input_size_voxels)
        self._output_size = self._voxel_size * \
            gp.Coordinate(output_size_voxels)

        self._checkpoint = checkpoint
        self._assemble_pipeline()

    def _assemble_pipeline(self):
        # define pipeline

        keys = {
            'RAW': gp.ArrayKey('RAW'),
            'LABELS': gp.ArrayKey('LABELS'),
            'MASK': gp.ArrayKey('MASK'),
            'METRIC_MASK': gp.ArrayKey('METRIC_MASK'),
            'PREDICTIONS': gp.ArrayKey('PREDICTIONS'),
        }

        self.request = gp.BatchRequest()
        self.request.add(keys['RAW'], self._input_size)
        self.request.add(keys['LABELS'], self._output_size)
        self.request.add(keys['MASK'], self._output_size)
        self.request.add(keys['METRIC_MASK'], self._output_size)
        self.request.add(keys['PREDICTIONS'], self._output_size)

        sources = fos.pipeline.sources.DataSourcesSemantic(
            config_file=self._data_config,
            keys=keys,
            data_path_prefix=self._data_path_prefix
        )

        if self._voxel_size != sources.voxel_size:
            raise ValueError((
                f"Specfied voxel size {self._voxel_size} does not match "
                f"the voxel size of the datasets {sources.voxel_size}."
            ))

        if sources.dataset_count > 1:
            raise NotImplementedError((
                f"Prediction is only implemented for a single cuboid ROI "
                f"at a time. "
                f"You provided {sources.dataset_count} in the data config "
                "file, please remove all but one."
            ))
        self.pipeline = sources.pipelines[0]

        # self.pipeline = (
        #     self.pipeline
        #     + fos.gunpowder.PadTo(keys['RAW'], self._input_size)
        #     + fos.gunpowder.PadTo(keys['LABELS'], self._output_size)
        #     + fos.gunpowder.PadTo(keys['MASK'], self._output_size)
        # )

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
            ])
        )

        predictions_roi = sources.rois[0].copy()
        # optionally grow the predictions roi to account for padded raw

        # labels
        predictions_roi = fos.utils.grow_roi_to(
            roi=predictions_roi,
            target_shape=self._output_size,
            voxel_size=self._voxel_size
        )
        logger.debug(f"{predictions_roi=}")

        self.predict = fos.gunpowder.torch.Predict(
            model=self._model,
            inputs={
                'x': keys['RAW'],
            },
            outputs={0: keys['PREDICTIONS']},
            array_specs={
                keys['PREDICTIONS']: gp.ArraySpec(
                    roi=predictions_roi,
                    dtype=np.float32,
                    voxel_size=self._voxel_size
                )
            },
            checkpoint=self._checkpoint,
            spawn_subprocess=True
        )

        self.pipeline = (
            self.pipeline
            + self.predict
        )

        self.pipeline = (
            self.pipeline
            + fos.gunpowder.Squeeze([
                keys['RAW'],
                keys['LABELS'],
                keys['MASK'],
                keys['PREDICTIONS'],
            ])
            + fos.gunpowder.Squeeze([keys['RAW']])
        )

        self.pipeline = (
            self.pipeline
            # scale back intensity to [0,1] interval
            + gp.IntensityScaleShift(keys['RAW'], 0.5, 0.5)
            + fos.gunpowder.ToDtype([keys['LABELS']], dtype='uint32')
        )

        self.pipeline = (
            self.pipeline
            + fos.gunpowder.Softmax(keys['PREDICTIONS'])
        )

        write_datasets = {}

        for cls in range(self._num_classes):
            pred_class_key = f"PREDICTIONS_CLASS_{cls}"
            keys[pred_class_key] = gp.ArrayKey(
                pred_class_key
            )
            self.request.add(keys[pred_class_key], self._output_size)
            self.pipeline = (
                self.pipeline
                + fos.gunpowder.PickChannel(
                    array=keys['PREDICTIONS'],
                    channel=cls,
                    output_array=keys[pred_class_key]
                )
            )
            write_datasets[keys[pred_class_key]] = \
                f"volumes/predictions/{self._run_id}/prob_maps/class_{cls}"

        keys['SEGMENTATION'] = gp.ArrayKey('SEGMENTATION')
        self.request.add(keys['SEGMENTATION'], self._output_size)
        write_datasets[keys['SEGMENTATION']
                       ] = f"volumes/predictions/{self._run_id}/segmentation"

        self.pipeline = (
            self.pipeline
            + fos.gunpowder.BinarizeLabels([keys['MASK']])
            + fos.gunpowder.ExtractSegmentation(
                array=keys['PREDICTIONS'],
                output_array=keys['SEGMENTATION'],
                mask=keys['MASK']
            )
        )

        self.pipeline = (
            self.pipeline
            + fos.gunpowder.ZarrWrite(
                dataset_names=write_datasets,
                # TODO change hacky access to filename
                output_filename=os.path.join(
                    os.path.expanduser(self._predictions_path_prefix),
                    sources.filenames[0]
                ),
                chunks=self._output_size / self._voxel_size / 2,
            )
        )

        self.scan = gp.Scan(self.request)
        self.pipeline = (
            self.pipeline
            + self.scan
        )
