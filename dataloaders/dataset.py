"""
This is self-defined dataset.

Created On 9th Mar, 2020
Author: Bohang Li
"""
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
from albumentations import Blur, JpegCompression, Compose
import random


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


class ConcatDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        samples = []
        real = []
        fake = []
        for path in os.listdir(image_folder):
            if has_file_allowed_extension(path, IMG_EXTENSIONS):
                label = path.split("_")[-1].split(".")[0]
                if label == "1":
                    fake.append(path)
                elif label == "0":
                    real.append(path)
                else:
                    print(label)
                    print(type(label))
                    raise ValueError("wrong label!")
        fake = random.sample(fake, len(real))
        for i in real:
            samples.append((i, 0))
        for j in fake:
            samples.append((j, 1))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        img_path, label = self.samples[item]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        label = [label] * 10
        sample = {"image": image, "label": label}
        return sample


class PatchDatasetFFHQ(Dataset):
    def __init__(self, image_name_list, img_folder_list, label_list=None, transform=None):
        self.image_name_list = image_name_list
        self.label_name_list = label_list
        self.img_folder_list = img_folder_list
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.img_folder_list[idx], self.image_name_list[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # trans = Compose([Blur(p=0.3), JpegCompression(p=0.2)])
        # image = trans(image=image)["image"]
        if self.transform:
            image = self.transform(image)
        if self.label_name_list is not None:
            label = self.label_name_list[idx]
        else:
            label = 0
        sample = {'image': image, 'label': label}
        return sample


class PatchDataset(Dataset):
    def __init__(self, image_name_list, img_folder, label_list=None, transform=None):
        self.image_name_list = image_name_list
        self.label_name_list = label_list
        self.img_folder = img_folder
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.img_folder, self.image_name_list[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # trans = Compose([Blur(p=0.3), JpegCompression(p=0.2)])
        # image = trans(image=image)["image"]
        if self.transform:
            image = self.transform(image)
        if self.label_name_list is not None:
            label = self.label_name_list[idx]
        else:
            label = 0
        sample = {'image': image, 'label': label}
        return sample
