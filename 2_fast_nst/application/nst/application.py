import torch
from . import utils
from torchvision import transforms


def stylize(image, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    image = content_transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image).cpu()
        output = utils.get_output_image(output[0])

    return output
