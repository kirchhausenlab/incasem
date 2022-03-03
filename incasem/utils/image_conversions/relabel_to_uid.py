import logging

import numpy as np
from .relabel_to import RelabelTo

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RelabelToUid(RelabelTo):
    """relabel all values not equal zero to the hash of
    the absolute path of the input folder

    """

    def __init__(self, dtype, input_path):

        value = hash(input_path)
        value_uint32 = np.array(value).astype('uint32')
        logger.info(
            f'Relabel non-zero values in {input_path} to {str(value_uint32)}')

        super().__init__(
            dtype=dtype,
            value=value
        )
