"""
This is face detection module. Initially, I use MTCNN as face detection model.
hint: the score of MTCNN might be a meta feature.
Input: Frames from videos.
Output: face patches.

Created On 22th Feb, 2020
Author: Bohang Li
"""
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import cv2
from PIL import Image, ImageDraw
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
from utils.logger import init_logging

if torch.cuda.is_available():
    device = 'cuda:0'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
print('Running on device: {0}'.format(device))


logger = init_logging(log_dir=os.path.abspath("../logs/"), log_file="face_detection.log")


def test_face_detection():
    """
    visually test on face detection.
    :return:
    """
    face_detector = MTCNN(margin=14, keep_all=True, factor=0.5, device=device).eval()
    img = cv2.imread("../testdata/uaspniazcl_000.jpg")
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    frame_draw = img.copy()
    draw = ImageDraw.Draw(frame_draw)
    boxes, scores = face_detector.detect(img)
    for box in boxes:
        draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
    frame_draw.save("../testdata/uaspniazcl_000_face.jpg")
    print(boxes)
    print(scores)


def _bbox_in_img(img, bbox):
    """
    check whether the bbox is inner an image.
    :param img: (3-d np.ndarray), image
    :param bbox: (list) [x, y, width, height]
    :return: (bool), whether bbox in image size.
    """
    if not isinstance(img, np.ndarray):
        raise ValueError("input image should be ndarray!")
    if len(img.shape) != 3:
        raise ValueError("input image should be (w,h,c)!")
    h = img.shape[0]
    w = img.shape[1]
    x_in = 0 <= bbox[0] <= w
    y_in = 0 <= bbox[1] <= h
    x1_in = 0 <= bbox[0] + bbox[2] <= w
    y1_in = 0 <= bbox[1] + bbox[3] <= h
    return x_in and y_in and x1_in and y1_in


def _enlarged_bbox(bbox, expand):
    """
    enlarge a bbox by given expand param.
    :param bbox: [x, y, width, height]
    :param expand: (tuple) (h,w), expanded pixels in height and width. if (int), same value in both side.
    :return: enlarged bbox
    """
    if isinstance(expand, int):
        expand = (expand, expand)
    s_0, s_1 = bbox[1], bbox[0]
    e_0, e_1 = bbox[1] + bbox[3], bbox[0] + bbox[2]
    x = s_1 - expand[1]
    y = s_0 - expand[0]
    x1 = e_1 + expand[1]
    y1 = e_0 + expand[0]
    width = x1 - x
    height = y1 - y
    return x, y, width, height


def _box_mode_cvt(bbox):
    """
    convert box from FCOS([xyxy], float) output to [x, y, width, height](int).
    :param bbox: (dict), an output from FCOS([x, y, x1, y1], float).
    :return: (list[int]), a box with [x, y, width, height] format.
    """
    if bbox is None:
        raise ValueError("There is no box in the dict!")
    # FCOS box format is [x, y, x1, y1]
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cvt_box = [int(bbox[0]), int(bbox[1]), int(w), int(h)]
    return cvt_box


def crop_bbox(img, bbox):
    """
    crop an image by giving exact bbox.
    :param img:
    :param bbox: [x, y, width, height]
    :return: cropped image
    """
    if not _bbox_in_img(img, bbox):
        raise ValueError("bbox is out of image size!img size: {0}, bbox size: {1}".format(img.shape, bbox))
    s_0 = bbox[1]
    s_1 = bbox[0]
    e_0 = bbox[1] + bbox[3]
    e_1 = bbox[0] + bbox[2]
    cropped_img = img[s_0:e_0, s_1:e_1, :]
    return cropped_img


def save_face_patches(frames_path: str, saving_path: str, expand_ratio=1.3):
    """
    get and save face patches from frames using MTCNN.
    :param frames_path: extracted frames path.
    :param saving_path: saving path of face patches.
    :param expand_ratio: bbox central enlarge ratio, default 1.3
    :return:
    """
    face_detector = MTCNN(margin=0, keep_all=True, thresholds=[0.7, 0.8, 0.8], factor=0.709, device=device).eval()
    patch_df = pd.DataFrame(columns=["PatchName", "Score"])
    df_score = []
    df_patch_name = []
    miss_bbox = []
    miss_face = []
    for file in tqdm(glob(os.path.join(frames_path, "*.jpg"))):
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        basename = str(os.path.basename(file).split(".")[0]) + "_face_"
        boxes, scores = face_detector.detect(img)
        if boxes is None:
            miss_face.append(file)
            continue
        for i, bbox in enumerate(boxes):
            patch_name = basename + str(i) + ".jpg"
            box_xywh = _box_mode_cvt(bbox)
            expand_w = int((box_xywh[2]*(expand_ratio-1))/2)
            expand_h = int((box_xywh[3]*(expand_ratio-1))/2)
            enlarged_box = _enlarged_bbox(box_xywh, (expand_h, expand_w))
            try:
                res = crop_bbox(img, enlarged_box)
            except ValueError:
                miss_bbox.append(file)
                continue
            score = scores[i]
            df_patch_name.append(patch_name)
            df_score.append(score)
            cv2.imwrite(os.path.join(saving_path, patch_name), res)
    patch_df["PatchName"] = df_patch_name
    patch_df["Score"] = df_score
    patch_df.to_csv(os.path.join(saving_path, "patch_image_statics.csv"), index=False)
    logger.info("miss bbox: {0}".format(set(miss_bbox)))
    logger.info("miss face: {0}".format(set(miss_face)))
    logger.info("miss bbox number: {0}".format(len(miss_bbox)))
    logger.info("miss face number: {0}".format(len(miss_face)))
    logger.info("total detect: {0} images".format(len(glob(os.path.join(frames_path, "*.jpg")))))


def get_face_patches(img, margin=10):
    """

    :param img: PIL Image, notice the RGB order if you open image with cv2.
    :param expand_ratio: enlarged bbox ratio, default 1.3
    :return:
    """
    face_detector = MTCNN(
        margin=margin, select_largest=False, keep_all=False,
        thresholds=[0.7, 0.8, 0.8], factor=0.709, device=device
    ).eval()
    faces, probs = face_detector(img, return_prob=True)
    return faces, probs


if __name__ == "__main__":
    save_face_patches(frames_path="../testdata", saving_path="../testdata")
