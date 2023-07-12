'''Base class for conversions of 2D slice images using Pillow and Numpy'''
from abc import ABC, abstractmethod


class ImageConversion(ABC):
    ''' TODO '''

    def __init__(self, dtype, **kwargs):
        """__init__.

        Args:
            dtype: numpy data type of output, if it is a numpy.ndarray
        """
        self.dtype = dtype

    @abstractmethod
    def __call__(self, img):
        """Convert single 2D image

        Args:
            img (PIL.Image):

        Returns:
            PIL.Image or numpy.ndarray
        """
