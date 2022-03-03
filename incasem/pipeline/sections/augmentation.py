import logging

import numpy as np
import gunpowder as gp

from .section import Section

logger = logging.getLogger(__name__)


class Augmentation(Section):
    """All data augmentations.

        Args:

            raw_key (gp.ArrayKey):

                The array that contains the raw EM data.

    """

    def __init__(self, raw_key):
        self._raw_key = raw_key
        super().__init__()

    # PROTECTED METHODS
    ###################

    def _define_nodes(self):
        """See base class.  """

        self._nodes['simple_0'] = gp.SimpleAugment()

        self._nodes['elastic'] = gp.ElasticAugment(
            control_point_spacing=(32, 32, 32),
            jitter_sigma=(2, 2, 2),
            # rotate around the leading axis
            rotation_interval=[0, 0.5 * np.pi],
            prob_slip=0.0,
            prob_shift=0.05,
            max_misalign=2,
            # higher subsample is faster, but deformation at worse resolution
            subsample=4
        )

        self._nodes['simple_1'] = gp.SimpleAugment()

        self._nodes['intensity'] = gp.IntensityAugment(
            array=self._raw_key,
            scale_min=0.9,
            scale_max=1.1,
            shift_min=-0.1,
            shift_max=0.1
        )

        # self._nodes['noise'] = gp.NoiseAugment(self._raw_key, var=0.0025)

    # PROPERTIES
    ############

    @property
    def raw_key(self):
        return self._raw_key
