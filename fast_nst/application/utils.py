"""
Utility file to convert the PIL image to OpenCV image, and convert it backward.
"""
import numpy as np
import cv2 as cv
from PIL import Image


def pil_to_opencv(image):
    """
    Convert PIL image to OpenCV image.

    :param image: The PIL image.
    :return: The OpenCV image.
    """
    i = np.array(image)
    red = i[:, :, 0].copy()
    i[:, :, 0] = i[:, :, 2].copy()
    i[:, :, 2] = red
    return i


def opencv_to_pil(image):
    """
    Convert OpenCV image to PIL image.

    :param image: The OpenCV image.
    :return: The PIL image.
    """
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return Image.fromarray(image)
