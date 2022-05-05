"""
The main entry for evaluating this multi-style model.
"""
import torch
import sys
from torchvision import transforms
from utility import *

# set device to gpu or cpu based on availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def stylize(image, model, input_style_id):
    """
    apply trained style transfer model to a content image given a style image we used during training
    """
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    image = content_transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image, style_id=[input_style_id]).cpu()
    transform_net_imshow(output[0], 'style' + str(input_style_id))
    save_image('./output/' +'_style'+str(input_style_id)+'.jpg', output[0])


if __name__ == '__main__':
    model_path =  sys.argv[1]
    output_path = sys.argv[2]
    style_num = int(sys.argv[3])
    content_image_path = sys.argv[4]
    transformer_net = TransformerNet(style_num=style_num).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    transformer_net.load_state_dict(checkpoint['model_state_dict'])
    content_image = transform_net_load_image(content_image_path)
    for i in range(style_num):
        stylize(content_image, transformer_net, i)
