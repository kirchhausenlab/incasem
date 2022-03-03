import logging

import numpy as np
import gunpowder as gp

from .data_sources_base import DataSourcesBase
from ...gunpowder.binarize_labels import BinarizeLabels
from ...gunpowder.merge_labels import MergeLabels
from ...gunpowder.add_background_labels import AddBackgroundLabels
from ...gunpowder.add_mask import AddMask
from ...gunpowder.merge_masks import MergeMasks

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataSourcesSemantic(DataSourcesBase):

    def _assemble_pipeline(
            self,
            attributes,
            file_path,
            key_suffix,
            voxel_size,
            roi,
            raw_roi=None
    ):
        datasets_gp = {}
        array_specs = {}

        if raw_roi is None:
            raw_roi = roi

        # process raw
        assert 'raw' in attributes
        datasets_gp[self._keys['RAW']] = attributes['raw']
        array_specs[self._keys['RAW']] = gp.ArraySpec(
            interpolatable=True,
            voxel_size=voxel_size
        )

        # process mask
        if 'mask' in attributes:
            datasets_gp[self._keys['MASK']] = attributes['mask']
            array_specs[self._keys['MASK']] = gp.ArraySpec(
                interpolatable=False,
                voxel_size=voxel_size
            )
        else:
            logger.info(f"No mask given, add dummy mask of all 1s.")

        # TODO calc on the fly
        # process metric masks
        if 'metric_masks' in attributes:
            array_keys_metric_mask = []
            for mm_ds in attributes['metric_masks']:
                mm_name = mm_ds.strip('/').split('/')[-1].upper()
                mm_key = gp.ArrayKey(f"METRIC_MASK_{key_suffix}_{mm_name}")

                datasets_gp[mm_key] = mm_ds
                array_specs[mm_key] = gp.ArraySpec(
                    interpolatable=False,
                    voxel_size=voxel_size
                )

                array_keys_metric_mask.append(mm_key)
        else:
            logger.info(
                f"No metric mask given, add dummy metric mask of all 1s.")

        # process labels

        if 'labels' in attributes:
            array_keys_labels = []
            classes = {}

            for ds, class_id in attributes['labels'].items():
                labels_name = ds.strip('/').split('/')[-1].upper()
                class_key = gp.ArrayKey(f'LABELS_{key_suffix}_{labels_name}')

                datasets_gp[class_key] = ds
                array_specs[class_key] = gp.ArraySpec(
                    interpolatable=False,
                    voxel_size=voxel_size
                )

                array_keys_labels.append(class_key)
                classes[class_key] = class_id

            assert np.all(
                sorted(classes.values()) == np.arange(len(classes)) + 1
            ), \
                f"Class labels {list(classes.values())} should be contiguous and start at 1."

            logger.debug(f'{classes=}')
        else:
            logger.info(f"No labels given, add dummy background labels.")

        pipeline = gp.ZarrSource(
            file_path,
            datasets=datasets_gp,
            array_specs=array_specs
        )
        for key, _ in array_specs.items():
            logger.debug(f'key: {key}')
            logger.debug(f'roi: {roi}')
            if key.__repr__() == 'RAW':
                roi_to_crop = raw_roi
            else:
                roi_to_crop = roi

            logger.debug(f'roi to crop: {roi_to_crop}\n')
            pipeline = (
                pipeline
                + gp.Crop(key, roi_to_crop)
            )

        if 'labels' in attributes:
            pipeline = (
                pipeline
                + BinarizeLabels(array_keys_labels)
            )

            pipeline = (
                pipeline
                + MergeLabels(
                    classes=classes,
                    output_array=self._keys['LABELS']
                )
            )
        else:
            pipeline = (
                pipeline
                + AddBackgroundLabels(
                    raw_array=self._keys['RAW'],
                    output_array=self._keys['LABELS']
                )
            )

        if 'mask' in attributes:
            pipeline = (
                pipeline
                + BinarizeLabels([self._keys['MASK']])
            )
        else:
            pipeline = (
                pipeline
                # TODO merge this and the AddLabelsNode
                + AddMask(
                    reference_array=self._keys['LABELS'],
                    output_array=self._keys['MASK']
                )
            )

        if 'metric_masks' in attributes:
            pipeline = (
                pipeline
                + BinarizeLabels(array_keys_metric_mask)
                + MergeMasks(
                    array_keys_metric_mask,
                    self._keys['METRIC_MASK']
                )
            )
        else:
            pipeline = (
                pipeline
                # TODO merge this and the AddLabelsNode
                + AddMask(
                    reference_array=self._keys['LABELS'],
                    output_array=self._keys['METRIC_MASK']
                )
            )

        pipeline = (
            pipeline
            + gp.Normalize(self._keys['RAW'])
        )

        return pipeline
