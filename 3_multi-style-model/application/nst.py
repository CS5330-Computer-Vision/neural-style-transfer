import torch
from torchvision import transforms
from utility import *

def stylize(image, model, input_style_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    style_num = 4
    model_path = '../training/model.pth'
    output_path = './output/'
    transformer_net = TransformerNet(style_num=style_num).to(device)
    checkpoint = torch.load(model_path)
    transformer_net.load_state_dict(checkpoint['model_state_dict'])
    content_image = transform_net_load_image('./eval/test.jpeg')
    for i in range(style_num):
        stylize(content_image, transformer_net, i)
