"""
The file includes the utility functions to get the pre-trained model, and show the output image from model.
"""
import torch

from PIL import Image
import matplotlib.pyplot as plt
from fast_nst.training import TransformerNet


def get_transformer_net_model(model_path):
    """
    Get the transformer network model.

    :param model_path: The path to the model parameters.
    :return: The model object
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerNet(padding_type='reflection', upsampling_type='nearest')
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def get_output_image(image):
    """
    Transform the given image to the displayable format.

    :param image: The original output image from model.
    :return: The displayable image.
    """
    with torch.no_grad():
        image = image.cpu().clone().clamp(0, 255).numpy()
        image = image.transpose(1, 2, 0).astype('uint8')
        image = Image.fromarray(image)

    return image


def transform_net_imshow(image, title=None):
    """
    Show the image from the model output.

    :param image: The output image.
    :param title: The title of the image to be shown.
    """
    image = get_output_image(image)
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.show()
