"""
The utility functions to load the image, show the image, do the transformation and save results locally.
"""

import torch
from PIL import Image
import matplotlib.pyplot as plt
from net import TransformerNet

# load image given a path
def transform_net_load_image(filename, size=None, scale=None):
  image = Image.open(filename).convert('RGB')
  target_size = image.size
  if size is not None:
    target_size = (size, size)
  elif scale is not None:
    target_size = (int(image.size[0] / scale), int(image.size[1] / scale))

  return image.resize(target_size, Image.ANTIALIAS)

# show image given a pth
def transform_net_imshow(image, title=None):
  with torch.no_grad():
    image = image.cpu().clone().clamp(0, 255).numpy()
    image = image.transpose(1, 2, 0).astype('uint8')
    image = Image.fromarray(image)

  fig = plt.figure(figsize=(5,5))
  plt.imshow(image)
  if title is not None:
    plt.title(title)
  plt.xticks([])
  plt.yticks([])
  plt.show()

# get model from the given path with the number of style images with which the model is trained
def get_transformer_net_model(model_path, style_num):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerNet(style_num=style_num)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model

# save image to the given path
def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)