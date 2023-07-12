import numpy as np
from .image_conversion import ImageConversion


class RelabelTo(ImageConversion):
    """relabel all values not equal zero to a given value"""

    def __init__(self, dtype, value, **kwargs):
        super().__init__(
            dtype=dtype,
            **kwargs
        )

        self.value = value

    def __call__(self, img):
        array = np.array(img, self.dtype)
        array[array != 0] = self.value
        return array
