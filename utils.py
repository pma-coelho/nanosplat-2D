import json
import torchvision
import os

def load_json(path):
    with open(path) as f:
        content = json.load(f)
    return content


def load_image(image_path):
    return torchvision.io.read_image(image_path).permute((1, 2, 0))


def save_image(image, path):
    folder, _ = os.path.split(path)
    if not os.path.exists(folder):
        os.mkdir(folder)
    torchvision.utils.save_image(image.permute((2, 0, 1)), path)