"""
Useful Transformers.

Created On 9th Mar, 2020
Author: bohang.li
"""
import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import Resize, RandomRotation, RandomHorizontalFlip, ToTensor, Normalize, ColorJitter, \
    ToPILImage
import skimage.transform as tsf
import random
import cv2

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
infer_transformer = transforms.Compose([
    ToPILImage(),
    Resize((300, 300)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

"""
Video-level transform.
"""


def get_birghtness(img):
    img_max = img.max()
    if img_max != 0:
        return img / img.max()
    else:
        return img


class RandomHFlip(object):
    def __init__(self, p, num_frame=10, size=240):
        self.num_frame = num_frame
        self.size = size
        self.p = p

    def __call__(self, frames):
        if random.random() < self.p:
            for x in range(self.num_frame):
                frames[:, x * self.size:(x + 1) * self.size, :] = cv2.flip(
                    frames[:, x * self.size:(x + 1) * self.size, :], 1)
            return frames
        return frames


class MinMaxNorm(object):
    def __init__(self, num_frame=10, size=240):
        self.num_frame = num_frame
        self.size = size

    def __call__(self, frames):
        for x in range(self.num_frame):
            frames[:, x * self.size:(x + 1) * self.size, :] = get_birghtness(
                frames[:, x * self.size:(x + 1) * self.size, :])
        return frames


class VideoToTensor(object):
    def __init__(self,  num_frame=10, size=240):
        self.num_frame = num_frame
        self.size = size

    def __call__(self, frames):
        # frames  # H, W, C
        tensor_list = []
        for x in range(self.num_frame):
            tensor_list.append(frames[:, x * self.size:(x + 1) * self.size, :])
        frames_tensor = torch.tensor(tensor_list)  # B, H, W, C
        return frames_tensor.permute(0, 3, 1, 2)  # B, C, H, W


def video_collate_fn(batch):
    data = [item["image"] for item in batch]
    target = [item["label"] for item in batch]
    data = torch.cat(data, dim=0)
    # target = torch.tensor(target).type(torch.LongTensor)
    target = torch.cat(target, dim=0)
    return [data, target]


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
