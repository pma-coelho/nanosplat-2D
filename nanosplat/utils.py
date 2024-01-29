import json
import torch
import torchvision
import os


def load_json(path):
    '''
    Load json file into python object.

    :param str path: Path to json file.

    :return: Python object (nested dictionary, list, ...)
    :rtype: object
    '''

    with open(path) as f:
        content = json.load(f)
    return content


def load_image(image_path, resize=None):
    '''
    Load image into torch tensor.

    :param str image_path: Path to image file.

    :return: Image tensor with shape (H, W, 3) and range [0, 1]
    :rtype: torch.tensor
    '''

    img = torchvision.io.read_image(image_path)
    if resize is not None and resize != -1:
        img = torchvision.transforms.Resize(resize, antialias=False)(img)
    return img.permute((1, 2, 0)) / 255


def save_image(image, path):
    '''
    Save torch tensor as image. Create directories if needed.

    :param torch.tensor image: Image tensor with shape (H, W, 3) and range [0, 1]
    :param str path: Image output file path.
    '''

    folder, _ = os.path.split(path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    torchvision.utils.save_image(image.permute((2, 0, 1)), path)