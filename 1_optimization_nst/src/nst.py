"""
Explored different layers of VGG-19 network for NST.
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
from torchvision.utils import save_image

from net import VGG_NST
from utility import *

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

# Executes style transfer for num_steps number of iterations to apply 
# style on content image 
def run_style_transfer(model, content_img, style_img, generated_img, 
                       num_steps=500, content_weight=1, style_weight=0.1, 
                       learning_rate=0.001, content_layer=2, style_layer_count=5,
                       init_image=0):
  # sets model to eval mode
  model.eval()
  # sets requires_grad to True for generated_img as the activation on the image
  # needs to be corrected with each iteration
  generated_img.requires_grad_(True)
  model.requires_grad_(False)
  optimizer = optim.Adam([generated_img], lr=learning_rate)

  # for logging purpose (between content/noise image as starting image)
  if init_image == 0:
    start_image = 'Content'
  else:
    start_image = 'Noise'

  print('=' * 50)
  print(f'Processing Image combination ==> Content layer: {content_layer}, Style layer before: {style_layer_count}, Initial Image: {start_image}')
  print('=' * 50)

  # gets features for content and style image, to be used in loss calculation
  content_features = model(content_img)
  style_features = model(style_img)
    
  for step in range(num_steps):
    # correct the values of updated input image
    with torch.no_grad():
      generated_img.clamp_(0, 1)

    # gets the convolution features for generated image
    generated_features = model(generated_img)

    style_loss = content_loss = 0
    # calculates content loss for the specified relu layer
    content_loss += get_content_loss(content_features[content_layer], generated_features[content_layer])

    count = 0
    # iterate through all the features for the chosen layers 
    for generated_feature, style_feature in zip(
      generated_features, style_features
      ):
      # calculates style loss for given layers
      if count < style_layer_count:
        style_loss += get_style_loss(style_feature, generated_feature)
        count += 1

    # gets total loss
    total_loss = content_weight * content_loss + style_weight * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if (step+1) % 100 == 0:
      print(f'Step: {step+1}/{num_steps} ==> Loss: {total_loss}')
    
  print(f'Final Image generated!!!')
  # returns final generated image
  return generated_img

# main function for executing task 1
def main(argv):
  style_img_path = argv[1]
  content_img_path = argv[2]
  num_steps = int(argv[3])

  # set device to cuda or cpu, based on availability 
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # for running tests on different combinations of content and style layer
  # sets style and content image
  style_img = image_loader(style_img_path, device)
  content_img = image_loader(content_img_path, device)

  assert style_img.size() == content_img.size(), \
      "we need to import style and content images of the same size"

  imshow(style_img, title='Style Image')
  imshow(content_img, title='Content Image')

  # sets vgg19 layers for relu1_2, relu2_2, relu3_3, relu4_3, relu5_3
  feature_layers = ["3", "8", "15", "24", "33"]
  model = VGG_NST(feature_layers).to(device)
  # hyperparameters
  # num_steps = 5000
  learning_rate = 0.01
  alpha = 1
  beta = 1000000

  # runs test for different content and style settings
  for content in range(5):
    for style in range(1, 6):
      for init in range(2):
        # sets contentimage or noise as starting target image
        if init == 0:
          generated_img = content_img.clone().to(device)
        else:
          generated_img = torch.randn(content_img.data.size(), device=device)
        
        # runs style loop
        output_img = run_style_transfer(model, content_img, style_img, generated_img, 
                                          num_steps, alpha, beta, learning_rate,
                                          content, style, init)
        
        # displays generated output image
        # imshow(output_img, title='Generated Image')
        img_name = "out_" + str(content) + "_" + str(style) + "_" + str(init) + ".png"
        save_image(output_img, img_name)
        print(f'Image {img_name} Saved!!\n')

# Checks if the code file is being executed
if __name__ == "__main__":
    main(sys.argv)