"""
The utility functions to load the image, show the image pair, and do the transformation.
"""
import torch
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


def transform_net_imshow_pair(original_image, output_image, original_title=None, output_title=None):
  """
    Show the image pair

    :param original_image: original content image
    :param output_image: output image
    :param original_title: the title of the original content image
    :param output_title: the title of the output image
  """
  with torch.no_grad():
    original_image = original_image.cpu().clone().clamp(0, 255).numpy()
    original_image = original_image.transpose(1, 2, 0).astype('uint8')
    original_image = Image.fromarray(original_image)

    output_image = output_image.cpu().clone().clamp(0, 255).numpy()
    output_image = output_image.transpose(1, 2, 0).astype('uint8')
    output_image = Image.fromarray(output_image)

  fig, ax = plt.subplots(1, 2, figsize = (10, 5))
  ax = ax.flatten()
  if original_title is not None:
    ax[0].set_title(original_title)
  ax[0].imshow(original_image)
  ax[0].set_axis_off()

  if output_title is not None:
    ax[1].set_title(output_title)
  ax[1].imshow(output_image)
  ax[1].set_axis_off()
  plt.show()

def gram_matrix(y):
  """
  Compute the gram matrix for the given image/image batch.
  :param y: The given image or image batch.
  """
  (b, ch, h, w) = y.size()
  features = y.view(b, ch, w * h)
  features_t = features.transpose(1, 2)
  gram = features.bmm(features_t) / (ch * h * w)
  return gram

def normalize_batch(batch):
  """
  Normalize the batch from the output of the transformer net,
  so that it fits the VGG input format, which assumes the image with values
  between 0 and 1.
  :param batch: The image batch to be normalized.
  """
  mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
  std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
  batch = batch.div_(255.0)
  return (batch - mean) / std