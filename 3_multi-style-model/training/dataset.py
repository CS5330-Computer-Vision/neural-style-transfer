"""
The customized class for the COCO dataset.
"""

import os

from torch.utils.data import Dataset
from utility import transform_net_load_image

class COCODataset(Dataset):
  """
  The customized dataset for the COCO data.
  """
  def __init__(self, image_directory, transform=None):
    """
    Initialize the COCODataset object with the given data directory, and transformer.
    :param image_directory: The directory to the images
    :param transform: The transformer
    """
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