import numpy as np
from PIL import Image

def resize_images(images, desired_dimensions):
    for name in images:
        im = Image.open(name)
        out = im.resize(desired_dimensions)
        out.save(f'formatted_{name}')

def rgb2gray(img_rgb):
    return img_rgb[:, :, 0] * 0.2989 + img_rgb[:, :, 1] * 0.5870 + img_rgb[:, :, 2] * 0.1140

def image_to_vector(filepath):
    img_data = np.asarray(Image.open(filepath))
    if len(img_data.shape) > 2:
        img_data = rgb2gray(img_data) / 255.0
    return img_data.reshape(-1)
