"""
This is self-defined dataset.

Created On 9th Mar, 2020
Author: Bohang Li
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from albumentations import Blur, JpegCompression, Compose
import random
import pickle


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


class FinalDataset(Dataset):
    def __init__(self, image_folder, kind, transform=None):
        if kind == "val":
            with open("../dataset/origin_map_val.pkl", "rb") as fp:
                data_dict = pickle.load(fp)
        else:
            with open("../dataset/origin_map_train.pkl", "rb") as fp:
                data_dict = pickle.load(fp)
        real = []
        fake = []
        for k, v in data_dict.items():
            real_path = os.path.join(image_folder, k)
            fake_candidates = random.sample(v, len(v))
            if not os.path.isfile(real_path):
                continue
            fake_path = os.path.join(image_folder, fake_candidates[0])
            for path in fake_candidates:
                fake_path = os.path.join(image_folder, path)
                if os.path.isfile(fake_path):
                    break
            if not os.path.isfile(fake_path):
                continue
            real.append(real_path)
            fake.append(fake_path)
        assert len(real) == len(fake)
        print("Real Sample:{0}, Fake Sample:{1}".format(len(real), len(fake)))
        self.samples = [(r, f) for r, f in zip(real, fake)]
        self.transform = transform

    def __getitem__(self, item):
        real_path, fake_path = self.samples[item]
        real = cv2.cvtColor(cv2.imread(real_path), cv2.COLOR_BGR2RGB)
        fake = cv2.cvtColor(cv2.imread(fake_path), cv2.COLOR_BGR2RGB)
        if self.transform:
            real = self.transform(real)
            fake = self.transform(fake)
        real = real.unsqueeze(0)
        fake = fake.unsqueeze(0)
        image = torch.cat((real, fake), dim=0)  # B, Time, C, H, W
        label = torch.tensor([0, 1])
        return image, label


class OriginDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        with open("../dataset/real_faces_1_1.pkl", "rb") as fp:
            real = pickle.load(fp)
        with open("../dataset/fake_faces_1_1.pkl", "rb") as fp:
            fake = pickle.load(fp)
        real_samples = [(os.path.join(image_folder, x), 0) for x in real]
        fake_samples = [(os.path.join(image_folder, x), 1) for x in fake]
        self.samples = real_samples + fake_samples

    def __getitem__(self, item):
        img_path, label = self.samples[item]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        # label = torch.tensor([label] * 10).type(torch.LongTensor)
        sample = {"image": image, "label": label}
        return sample


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
                    fake.append(os.path.join(image_folder, path))
                elif label == "0":
                    real.append(os.path.join(image_folder, path))
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
        # label = torch.tensor([label] * 10).type(torch.LongTensor)
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
