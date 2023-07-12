from PIL import ImageOps
from .image_conversion import ImageConversion


class RgbToGrayInversion(ImageConversion):
    """ First, convert from RGB to grayscale, then invert the grayscale"""

    def __call__(self, img):
        img = img.convert('L')
        img = ImageOps.invert(img)
        return img
