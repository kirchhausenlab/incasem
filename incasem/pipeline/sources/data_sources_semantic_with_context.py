import logging
import os
import json

import gunpowder as gp

from .data_sources_semantic import DataSourcesSemantic

logger = logging.getLogger(__name__)


class DataSourcesSemanticWithContext(DataSourcesSemantic):

    def __init__(self, context, **kwargs):
        self.context = context
        super().__init__(**kwargs)

    def _assemble_pipeline(
            self,
            attributes,
            file_path,
            key_suffix,
            voxel_size,
            roi,
    ):

        assert 'raw' in attributes

        # get the raw datasets to ensure sufficient context
        raw_path = os.path.join(
            file_path,
            attributes['raw']
        )
        with open(os.path.join(raw_path, '.zarray')) as f:
            raw_arr_meta = json.load(f)
            raw_shape = gp.Coordinate(raw_arr_meta['shape']) * voxel_size
        with open(os.path.join(raw_path, '.zattrs')) as f:
            raw_attrs_meta = json.load(f)
            raw_offset = gp.Coordinate(raw_attrs_meta['offset'])

        raw_max_roi = gp.Roi(offset=raw_offset, shape=raw_shape)
        logger.debug(f'raw max roi: {raw_max_roi}')

        context_voxel = gp.Coordinate(self.context) * voxel_size
        raw_roi = roi.grow(context_voxel, context_voxel)
        logger.debug(f'raw roi: {raw_roi}')

        raw_available_roi = raw_max_roi.intersect(raw_roi)
        logger.debug(f'raw intersect roi: {raw_available_roi}')

        return super()._assemble_pipeline(
            attributes,
            file_path,
            key_suffix,
            voxel_size,
            roi,
            raw_roi=raw_available_roi)
