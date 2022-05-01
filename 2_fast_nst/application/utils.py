import numpy as np
import torch
import torchvision
import cv2 as cv
from PIL import Image


def pil_to_opencv(image):
    i = np.array(image)
    red = i[:, :, 0].copy()
    i[:, :, 0] = i[:, :, 2].copy()
    i[:, :, 2] = red
    return i


def opencv_to_pil(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return Image.fromarray(image)
