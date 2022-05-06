"""
Utility functions for NST.
"""

# import statements
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torchvision import transforms, models

# gets content loss between content image and target image
def get_content_loss(content, target):
  # gets mean square error between content and target vector
  loss = F.mse_loss(content, target)
  return loss

# creates gram matrix for tensor
def gram_matrix(input):
  b, c, h, w = input.size() # shape: (batch, channel, height, width)
  features = input.view(b*c, h*w) 
  # computes gram matrix product 
  G = torch.mm(features, features.t()) 
  # normalize the values 
  return G.div(b*c*h*w)

# gets style loss between style image and target image
def get_style_loss(style, target):
  # gets gram matrix for both style and target tensors
  S = gram_matrix(style)
  T = gram_matrix(target)

  # gets mean square error between style and target vector
  loss = F.mse_loss(S, T)
  return loss

# loads image as tensor
def image_loader(image_name, device):
  # desired size of the output image
  imsize = 512 if torch.cuda.is_available() else 256 
  loader = transforms.Compose(
    [
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor()
    ]
  )
  image = Image.open(image_name)
  # insert batch dimension 
  image = loader(image).unsqueeze(0)
  return image.to(device, torch.float)

# displays tensor as image
def imshow(tensor, title=None):
  unloader = transforms.Compose(
    [
        transforms.ToPILImage()  # reconvert into PIL image
    ]
  )

  # clone tensor and move to cpu
  image = tensor.cpu().clone()  
  # removes batch dimension
  image = image.squeeze(0)      
  image = unloader(image)

  fig = plt.figure(figsize=(5,5))
  plt.imshow(image)
  if title is not None:
    plt.title(title)
  plt.xticks([])
  plt.yticks([])
  plt.show()