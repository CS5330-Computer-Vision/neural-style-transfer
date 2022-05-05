"""
The main entry for the training project.
"""
import os

import torch
from torchvision import transforms
from .net import TransformerNet, Vgg19
from torch.optim import Adam
from torch.utils.data import DataLoader

# set device to gpu or cpu based on availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)


def train(style_image_path="./rain-princess-cropped.jpeg",
          dataset_path="./train2017",
          image_size=256,
          epoch_size=2,
          batch_size=4,
          learning_rate=1e-3,
          content_weight=1e5,
          style_weight=1e10,
          testing=False,
          padding_type='reflection',
          upsampling_type='nearest'
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

    # initialize the transformer net
    transformer_net = TransformerNet(padding_type, upsampling_type).to(device)
    optimizer = Adam(transformer_net.parameters(), learning_rate)
    mse_loss = torch.nn.MSELoss()

    # initialize the vgg-19 net
    vgg = Vgg19().to(device)

    # prepare the style image as the input
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style_image = transform_net_load_image(style_image_path, size=image_size)
    style_image = style_transform(style_image)
    # repeat the style image multiple times to fit the size of the batch data
    style_image = style_image.repeat(batch_size, 1, 1, 1).to(device)

    # extract the features from the style image
    style_image_vgg_output = vgg(normalize_batch(style_image))
    style_features = [gram_matrix(y) for y in style_image_vgg_output]

    # statistics
    content_loss_list = []
    style_loss_list = []
    image_count_index = []
    total_image_count = 0

    for epoch in range(epoch_size):
        transformer_net.train()
        average_content_loss = 0.
        average_style_loss = 0.
        image_count = 0

        for batch_id, (x, _) in enumerate(train_loader):
            batch_size = len(x)
            image_count += batch_size
            total_image_count += batch_size
            optimizer.zero_grad()

            x = x.to(device)
            original_image = x.cpu().clone()
            output_image = transformer_net(x)
            original_output_image = output_image.cpu().clone()

            output_image = normalize_batch(output_image)
            x = normalize_batch(x)

            output_image_vgg_outputs = vgg(output_image)
            features_x = vgg(x)

            # get the content loss
            content_loss = content_weight * mse_loss(features_x[2],
                                                     output_image_vgg_outputs[2])

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

            if testing:
                content_loss_list.append(content_loss.item())
                style_loss_list.append(style_loss.item())
                image_count_index.append(total_image_count)

            if (batch_id + 1) % 500 == 0:
                print(
                    f'Epoch {epoch}:\t[{image_count}/{len(train_dataset)}]\taverage content loss: {average_content_loss / (batch_id + 1)}\taverage style loss: {average_style_loss / (batch_id + 1)}\ttotal loss: {(average_content_loss + average_style_loss) / (batch_id + 1)}')
                transform_net_imshow(original_output_image[0], title='output image')
                transform_net_imshow(original_image[0], title='content image')

                if not testing:
                    content_loss_list.append(content_loss.item())
                    style_loss_list.append(style_loss.item())
                    image_count_index.append(total_image_count)
                else:
                    break

        model_name = f'epoch_{epoch}.model'
        model_path = os.path.join('./drive/MyDrive/model', model_name)
        torch.save(transformer_net.state_dict(), model_path)

    plt.plot(image_count_index, content_loss_list)
    plt.plot(image_count_index, style_loss_list)
    plt.legend(['content loss', 'style loss'])
    plt.xlabel('image count')
    plt.show()
