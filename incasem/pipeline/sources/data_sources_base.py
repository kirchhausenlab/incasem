import logging
import os
import json
from abc import ABC, abstractmethod

import gunpowder as gp

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataSourcesBase(ABC):
    def __init__(self, config_file, keys, data_path_prefix):

        self._config_file = os.path.expanduser(config_file)
        self._keys = keys
        self._data_path_prefix = data_path_prefix

        self._pipeline = None
        self._dataset_count = 0
        self._names = []
        self._filenames = []
        self._rois = []
        self._voxel_size = None

        self.pipelines = self._assemble()

    # PROTECTED METHODS
    ###################

    @abstractmethod
    def _assemble_pipeline(
            self, attributes, file_path, key_suffix, voxel_size, roi):
        pass

    def _assemble(self):
        # parse config file
        with open(self._config_file, 'r') as f:
            data_sources = json.load(f)

        # process each dataset
        pipelines = []
        for name, attributes in data_sources.items():
            file_path, key_suffix, voxel_size, roi = self._parse_metadata(
                name=name,
                attributes=attributes
            )
            pl = self._assemble_pipeline(
                attributes, file_path, key_suffix, voxel_size, roi)
            pipelines.append(pl)
            self._dataset_count += 1
            self._names.append(name)
            self._filenames.append(attributes['file'])

        logger.debug(f'{len(pipelines)=}')
        return pipelines

    def _parse_metadata(self, name, attributes):
        logger.info(f"Setting up {name}")

        # prepare key suffix
        key_suffix = name.strip().upper()

        # parse file path
        assert 'file' in attributes
        file_path = os.path.expanduser(
            os.path.join(self._data_path_prefix, attributes['file'])
        )

        # parse voxel_size, offset, shape
        try:
            voxel_size = gp.Coordinate(attributes['voxel_size'])
            logger.debug(f"{voxel_size=}")
        except KeyError:
            raise ValueError(
                f"Voxel size for {name} not specified in data config."
            )
        self._check_voxel_size(voxel_size)

        try:
            offset = gp.Coordinate(attributes['offset']) * voxel_size
            logger.debug(f"{offset=}")
            shape = gp.Coordinate(attributes['shape']) * voxel_size
            logger.debug(f"{shape=}")
            roi = gp.Roi(offset=offset, shape=shape)
            self._rois.append(roi)
            logger.debug(f"{roi=}")
        except KeyError:
            logger.warning((
                f"Offset/shape for {name} not specified, "
                "therefore not setting a ROI"
            ))
            roi = None

        return file_path, key_suffix, voxel_size, roi

    def _check_voxel_size(self, voxel_size):
        """Ensure that all datasets have the same voxel size."""
        if self._voxel_size is None:
            self._voxel_size = voxel_size
        else:
            if self._voxel_size != voxel_size:
                raise ValueError((
                    "The datasets have different voxel sizes, "
                    "unable to proceed."
                ))

    # PROPERTIES
    ############

    @property
    def config_file(self):
        return self._config_file

    @property
    def keys(self):
        return self._keys

    @property
    def data_path_prefix(self):
        return self._data_path_prefix

    @property
    def pipeline(self):
        return self._pipeline

    @property
    def dataset_count(self):
        return self._dataset_count

    @property
    def names(self):
        return self._names

    @property
    def filenames(self):
        return self._filenames

    @property
    def rois(self):
        return self._rois

    @property
    def voxel_size(self):
        return self._voxel_size
