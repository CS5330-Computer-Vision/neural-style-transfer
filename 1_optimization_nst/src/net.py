"""
VGG-19 network customized for NST.
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

# NST VGG model
class VGG_NST(nn.Module):
  def __init__(self, features):
    super(VGG_NST, self).__init__()
    # Sets chosen feature to passed features
    self.chosen_features = features

    # we don't need to run anything further than relu5_3 layer
    self.model = models.vgg19(pretrained=True).features[:34]

  def forward(self, x):
    # stores relevant features from output
    features = []

    # Go through each layer in model and store the output of the layer 
    # in features, if the layer is in the chosen_features. 
    # Return all the activations for the specific layers in chosen_features.
    for layer_num, layer in enumerate(self.model):
      x = layer(x)
      if str(layer_num) in self.chosen_features:
        features.append(x)

    return features