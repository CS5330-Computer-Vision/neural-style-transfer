"""
The file includes all the network model that will be used.
"""
import torch
from torchvision import models
from collections import namedtuple


# VGG-19 model for NST
class Vgg19(torch.nn.Module):
  def __init__(self, requires_grad=False):
    super(Vgg19, self).__init__()
    # use relu1_2, relu2_2, relu3_3, relu4_3 as mentioned in NST paper
    self.chosen_features = ['3', '8', '15', '24']

    # we don't need to run anything further than conv5_1 layer
    self.model = models.vgg19(pretrained=True).features[:25]

    if not requires_grad:
      for param in self.parameters():
        param.requires_grad = False

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

    vgg_outputs = namedtuple("vgg_outputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
    return vgg_outputs(features[0], features[1], features[2], features[3])
    # return features

class ConvLayer(torch.nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride):
    super(ConvLayer, self).__init__()
    reflection_padding = kernel_size // 2
    self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
    self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

  def forward(self, x):
    x = self.reflection_pad(x)
    x = self.conv2d(x)
    return x


class ResidualBlock(torch.nn.Module):
  def __init__(self, channels):
    super(ResidualBlock, self).__init__()
    self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
    self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
    self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
    self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
    self.relu = torch.nn.ReLU()

  def forward(self, x):
    residual = x
    x = self.relu(self.in1(self.conv1(x)))
    x = self.in2(self.conv2(x))
    x = x + residual
    return x


class UpsampleConvLayer(torch.nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
    super(UpsampleConvLayer, self).__init__()
    self.upsample = upsample
    reflection_padding = kernel_size // 2
    self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
    self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

  def forward(self, x):
    if self.upsample:
      x = torch.nn.functional.interpolate(x, mode='nearest', scale_factor=self.upsample)
    x = self.reflection_pad(x)
    return self.conv2d(x)


class BatchInstanceNorm2d(torch.nn.Module):
  def __init__(self, style_num, in_channels):
    super(BatchInstanceNorm2d, self).__init__()
    self.inns = torch.nn.ModuleList([torch.nn.InstanceNorm2d(in_channels, affine=True) for i in range(style_num)])

  def forward(self, x, style_id):
      out = torch.stack([self.inns[style_id[i]](x[i].unsqueeze(0)).squeeze_(0) for i in range(len(style_id))])
      return out


class TransformerNet(torch.nn.Module):
  def __init__(self, style_num):
    super(TransformerNet, self).__init__()
    self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
    self.in1 = BatchInstanceNorm2d(style_num, 32)
    self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
    self.in2 = BatchInstanceNorm2d(style_num, 64)
    self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
    self.in3 = BatchInstanceNorm2d(style_num, 128)
    # residual layers
    self.res1 = ResidualBlock(128)
    self.res2 = ResidualBlock(128)
    self.res3 = ResidualBlock(128)
    self.res4 = ResidualBlock(128)
    self.res5 = ResidualBlock(128)
    # upsampling layers
    self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
    self.in4 = BatchInstanceNorm2d(style_num, 64)
    self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
    self.in5 = BatchInstanceNorm2d(style_num, 32)
    self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)

    self.relu = torch.nn.ReLU()

  def forward(self, x, style_id):
    x = self.relu(self.in1(self.conv1(x), style_id))
    x = self.relu(self.in2(self.conv2(x), style_id))
    x = self.relu(self.in3(self.conv3(x), style_id))
    x = self.res1(x)
    x = self.res2(x)
    x = self.res3(x)
    x = self.res4(x)
    x = self.res5(x)
    x = self.relu(self.in4(self.deconv1(x), style_id))
    x = self.relu(self.in5(self.deconv2(x), style_id))
    return self.deconv3(x)