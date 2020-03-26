import os
import cv2
from albumentations import Blur, JpegCompression, Compose
from glob import glob
from tqdm import tqdm

raw_image_path = "/home/chongmin/karkin/git_files/anti-deepfake/dataset/videos"


def hflip(frames):
    for x in range(10):
        frames[:, x * 240:(x + 1) * 240, :] = cv2.flip(
            frames[:, x * 240:(x + 1) * 240, :], 1)
    return frames


def blur(frames):
    # trans = Compose([Blur(p=1), JpegCompression(p=1)])
    trans = Compose([JpegCompression(p=1, quality_lower=70, quality_upper=90)])
    for x in range(10):
        frames[:, x * 240:(x + 1) * 240, :] = trans(image=frames[:, x * 240:(x + 1) * 240, :])["image"]
    return frames


for img in tqdm(glob(os.path.join(raw_image_path, "*.jpg"))):
    basename = os.path.basename(img).split(".")[0] + "_blur.jpg"
    image = cv2.imread(img)
    new_image = blur(image)
    cv2.imwrite(os.path.join("../dataset/videos_aug/blur", basename), new_image)
