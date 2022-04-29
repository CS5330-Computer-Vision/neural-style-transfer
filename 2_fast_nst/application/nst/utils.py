import os
import sys
import time
import re
from collections import namedtuple

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt


def transform_net_load_image(filename, size=None, scale=None):
    """
    Load the image by the given filename. The image would be resized if size or scale is set.

    :param size: New size of the image
    :param scale: The scale ratio to resize the image
    :return: The resized image
    """
    image = Image.open(filename).convert('RGB')
    target_size = image.size
    if size is not None:
        target_size = (size, size)
    elif scale is not None:
        target_size = (int(image.size[0] / scale), int(image.size[1] / scale))

    return image.resize(target_size, Image.ANTIALIAS)


def get_output_image(image):
    with torch.no_grad():
        image = image.cpu().clone().clamp(0, 255).numpy()
        image = image.transpose(1, 2, 0).astype('uint8')
        image = Image.fromarray(image)

    return image


def transform_net_imshow(image, title=None):
    image = get_output_image(image)
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std


class COCODataset(Dataset):
    def __init__(self, image_directory, transform=None):
        self.image_names = []
        self.image_directory = image_directory
        self.transform = transform
        for image_name in os.listdir(image_directory):
            self.image_names.append(image_name)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = os.path.join(self.image_directory, self.image_names[index])
        image = transform_net_load_image(image_name)

        if self.transform:
            image = self.transform(image)

        return image, 0
