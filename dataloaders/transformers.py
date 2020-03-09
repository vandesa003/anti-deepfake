"""
Useful Transformers.

Created On 9th Mar, 2020
Author: bohang.li
"""
import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import Resize, RandomRotation, RandomHorizontalFlip, ToTensor, Normalize, ColorJitter, ToPILImage
import skimage.transform as tsf


train_transformer = transforms.Compose([
    ToPILImage(),
    Resize((300, 300)),
    RandomHorizontalFlip(p=0.3),
    RandomRotation(degrees=15),
    ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


val_transformer = transforms.Compose([
    ToPILImage(),
    Resize((300, 300)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# TODO: infer_transform !!!

"""
Segmentation Transformer/Pixel-level task transformer.
"""


class Rescale(object):
    """
    Rescale for ndarray.
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, image):
        img = tsf.resize(image, (self.height, self.width), mode='constant')
        return img


class ToTensor(object):
    """
    convert narrays in sample to Tensors.
    """
    def __call__(self, image):
        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
        # Convert image from [0, 255] to [0, 1].
        image = image / np.max(image)
        if image.shape[2] == 1:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
        # change the r,g,b to b,r,g from [0,255] to [0,1]
        # transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        tmpImg = tmpImg.transpose((2, 0, 1))
        return torch.from_numpy(tmpImg)



