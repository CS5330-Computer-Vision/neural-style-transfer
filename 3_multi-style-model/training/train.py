# -*- coding: utf-8 -*-
"""
This file is the main entry for training the multi-style model
"""

import os
import torch
import sys
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from dataset import COCODataset
from utility import *
from net import TransformerNet, Vgg19

# set device to gpu or cpu based on availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(path = 'model.pth',
          style_folder_path="image/", 
          dataset_path="./train2017", 
          image_size=256,
          style_size=512, 
          epoch_size=2,
          batch_size=4, 
          learning_rate=1e-3,
          content_weight=1e5,
          style_weight=1e10
          ):
  # load the training data
  data_transform = transforms.Compose([
                                     transforms.Resize(image_size),
                                     transforms.CenterCrop(image_size),
                                     transforms.ToTensor(),
                                     transforms.Lambda(lambda x: x.mul(255))
  ])
  train_dataset = COCODataset(dataset_path, data_transform)
  train_loader = DataLoader(train_dataset, batch_size=batch_size)

  # load style images to train
  style_image_names = [f for f in os.listdir(style_folder_path)]
  style_num = len(style_image_names)

  # initialize the transformer net
  transformer_net = TransformerNet(style_num=style_num).to(device)
  optimizer = Adam(transformer_net.parameters(), learning_rate)
  mse_loss = torch.nn.MSELoss()

  vgg = Vgg19().to(device)
  style_transform = transforms.Compose([
                                        transforms.Resize(style_size),
                                        transforms.CenterCrop(style_size),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.mul(255))
  ])

  style_batch = []
  for i in range(style_num):
    style_image = transform_net_load_image(style_folder_path + style_image_names[i], size=style_size)
    style_image = style_transform(style_image)
    style_batch.append(style_image)

  style_image = torch.stack(style_batch).to(device)

  # extract the features from the style image
  style_image_vgg_output = vgg(normalize_batch(style_image))
  style_features = [gram_matrix(y) for y in style_image_vgg_output]

  # statistics
  content_loss_list = []
  style_loss_list = []
  image_count_index = []
  total_image_count = 0
  epoch_start = 0

  # Load pre-trained model
  checkpoint = torch.load(path)
  transformer_net.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  content_loss_list = checkpoint['content_loss']
  style_loss_list = checkpoint['style_loss']
  image_count_index = checkpoint['count_index']
  total_image_count = image_count_index[-1]
  epoch_start = checkpoint['epoch']

  for epoch in range(epoch_start, epoch_size):
    transformer_net.train()
    average_content_loss = 0.
    average_style_loss = 0.
    image_count = 0

    for batch_id, (x, _) in enumerate(train_loader):
      cur_batch_size = len(x)

      if cur_batch_size < batch_size:
          break # skip to next epoch when no enough images left in the last batch of current epoch
      image_count += cur_batch_size
      total_image_count += cur_batch_size
      optimizer.zero_grad()

      batch_style_id = [i % style_num for i in range(image_count-cur_batch_size, image_count)]
      x = x.to(device)
      original_image = x.cpu().clone()
      output_image = transformer_net(x, style_id=batch_style_id)
      original_output_image = output_image.cpu().clone()
      output_image = normalize_batch(output_image)
      x = normalize_batch(x)

      output_image_vgg_outputs = vgg(output_image)
      features_x = vgg(x)

      # get the content loss
      content_loss = content_weight * mse_loss(output_image_vgg_outputs.relu3_3, features_x.relu3_3)

      # get the style loss
      style_loss = 0.
      for output_image_vgg_output, style_feature in zip(output_image_vgg_outputs, style_features):
        output_image_style_feature = gram_matrix(output_image_vgg_output)
        style_loss += mse_loss(output_image_style_feature, style_feature[:batch_size, :, :])
      style_loss *= style_weight

      # run the optimization
      total_loss = content_loss + style_loss
      total_loss.backward()
      optimizer.step()

      # calculate the statistic
      average_content_loss += content_loss.item()
      average_style_loss += style_loss.item()

      if (batch_id + 1) % 1000 == 0:
        print(f'Epoch {epoch}:\t[{image_count}/{len(train_dataset)}]\taverage content loss: {average_content_loss / (batch_id + 1)}\taverage style loss: {average_style_loss / (batch_id + 1)}\ttotal loss: {(average_content_loss + average_style_loss) / (batch_id + 1)}')
        for i in range(len(original_output_image)):
          transform_net_imshow_pair(original_image[i], original_output_image[i], 'Content Image: ' + str(i+1), 'Output Image: ' + str(i+1))

        content_loss_list.append(content_loss.item())
        style_loss_list.append(style_loss.item())
        image_count_index.append(total_image_count)

        # saves the model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': transformer_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'content_loss': content_loss_list,
            'style_loss': style_loss_list,
            'count_index': image_count_index
            }, path)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': transformer_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'content_loss': content_loss_list,
        'style_loss': style_loss_list,
        'count_index': image_count_index
        }, path)

  plt.plot(image_count_index, content_loss_list)
  plt.plot(image_count_index, style_loss_list)
  plt.legend(['content loss', 'style loss'])
  plt.xlabel('Number of training images seen')
  plt.ylabel('Loss')
  plt.title('Model Loss')
  print(image_count_index)
  plt.show()

if __name__ == '__main__':
  model_path = sys.argv[1]
  style_folder_path = sys.argv[2]
  dataset_path = sys.argv[3]

  train(path=model_path, style_folder_path=style_folder_path, dataset_path=dataset_path)