"""
The file includes all the network model that will be used.
"""
import torch
from torch import nn
from torchvision import models


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

        return features


class ConvLayer(torch.nn.Module):
    """
  Convolution layer to retain the same size of the image by padding.
  Different padding methods are supported.
  """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding_type):
        """
    Initialize the ConvLayer object with the same parameters as the nn.Conv2d.

    :param in_channels: The number of input channels
    :param out_channels: The number of output channels
    :param kernel_size: The size of the kernel. It should be an integer, which
      represents the height and width of the square kernel
    :param stride: The stride for the ConvLayer
    :param padding_type: The type for the padding. Should be one of ['zeor',
      'reflection', 'replication']
    """
        super(ConvLayer, self).__init__()
        if padding_type == 'zero':
            self.pad2d = nn.ZeroPad2d(kernel_size // 2)
        elif padding_type == 'reflection':
            self.pad2d = nn.ReflectionPad2d(kernel_size // 2)
        elif padding_type == 'replication':
            self.pad2d = nn.ReplicationPad2d(kernel_size // 2)

        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                      stride)

    def forward(self, x):
        x = self.pad2d(x)
        x = self.conv2d(x)
        return x


class ResidualBlock(torch.nn.Module):
    """
  Referenced from http://torch.ch/blog/2016/02/04/resnets.html
  """

    def __init__(self, channels, padding_type):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1,
                               padding_type=padding_type)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1,
                               padding_type=padding_type)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.in1(self.conv1(x)))
        x = self.in2(self.conv2(x))
        x = x + residual
        return x


class UpsampleConvLayer(torch.nn.Module):
    """
    Upsample ConvLayer to upscale the image with the given factor.
    Different upsampling methods are supported.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample,
                 padding_type, upsampling_type):
        """
    Initialize the Upsample ConvLayer object with the parameters of Conv2d,
    and the paramters of interpolate function.
    """
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        self.upsampling_type = upsampling_type
        self.conv2d = ConvLayer(in_channels, out_channels, kernel_size, stride,
                                padding_type)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, mode=self.upsampling_type,
                                            scale_factor=self.upsample)
        return self.conv2d(x)


class TransformerNet(torch.nn.Module):
    def __init__(self, padding_type, upsampling_type):
        """
    Initialize the Transformer Net object by conv layers, residual blocks, and
    upsampling layers.
    """
        super(TransformerNet, self).__init__()

        # construct the conv layers to embed the input image into lower dimensions
        self.conv_layers = nn.Sequential(
            ConvLayer(3, 32, kernel_size=9, stride=1, padding_type=padding_type),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            ConvLayer(32, 64, kernel_size=3, stride=2, padding_type=padding_type),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            ConvLayer(64, 128, kernel_size=3, stride=2, padding_type=padding_type),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
        )

        # construct the residual blocks
        self.residual_blocks = nn.Sequential(
            ResidualBlock(128, padding_type),
            ResidualBlock(128, padding_type),
            ResidualBlock(128, padding_type),
            ResidualBlock(128, padding_type),
            ResidualBlock(128, padding_type)
        )

        # construct the upsampling layers
        self.upsampling_layers = nn.Sequential(
            UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2,
                              padding_type=padding_type,
                              upsampling_type=upsampling_type),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2,
                              padding_type=padding_type,
                              upsampling_type=upsampling_type),
            nn.InstanceNorm2d(32, affine=True),
            ConvLayer(32, 3, kernel_size=9, stride=1, padding_type=padding_type)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.residual_blocks(x)
        x = self.upsampling_layers(x)
        return x
