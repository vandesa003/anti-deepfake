"""
This is face detection module. Initially, I use MTCNN as face detection model.
hint: the score of MTCNN might be a meta feature.
Input: Frames from videos.
Output: face patches.

Created On 22th Feb, 2020
Author: Bohang Li
"""
import os
import sys
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(dir_name, "../"))
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
    cvt_box = [int(bbox[0]), int(bbox[1]), max(int(w), 0), max(int(h), 0)]
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


def save_face_patches(frames_paths: list, saving_path: str, expand_ratio=1.3, batch=None, gpu_id=None):
    """
    get and save face patches from frames using MTCNN.
    :param frames_paths: list, extracted frames paths.
    :param saving_path: saving path of face patches.
    :param expand_ratio: bbox central enlarge ratio, default 1.3
    :param batch: used for multi-processing.
    :param gpu_id: used for multi-processing.
    :return:
    """
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
    hard_samples = []
    face_detector = MTCNN(margin=0, keep_all=False, select_largest=False, thresholds=[0.7, 0.8, 0.8],
                          min_face_size=60, factor=0.8, device=device).eval()
    patch_df = pd.DataFrame(columns=["PatchName", "Score"])
    df_score = []
    df_patch_name = []
    for file in tqdm(frames_paths):
        img = cv2.imread(file)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        basename = str(os.path.basename(file).split(".")[0]) + "_face_0.jpg"
        boxes, confidences = face_detector.detect(img)
        if boxes is None:
            hard_samples.append(basename.split("_")[0])
            continue
        best_confidence = confidences[0]
        best_box = boxes[0, :]
        best_face = face_boxes_post_process(img, best_box, expand_ratio=expand_ratio)
        df_score.append(best_confidence)
        df_patch_name.append(basename)
        cv2.imwrite(os.path.join(saving_path, basename), best_face)

    patch_df["PatchName"] = df_patch_name
    patch_df["Score"] = df_score
    hard_samples = set(hard_samples)
    hard_samples = pd.DataFrame(hard_samples, columns=["hashes"])
    if batch is None:
        patch_df.to_csv(os.path.join(saving_path, "patch_image_statics.csv"), index=False)
        hard_samples.to_csv("hard_samples.csv", index=False)
    else:
        patch_df.to_csv(os.path.join(saving_path, "patch_image_statics_batch_{0}.csv".format(batch)), index=False)
        hard_samples.to_csv("hard_samples.csv".format(batch), index=False)


def crop_resize(img, box, image_size):
    """
    learn from MTCNN.
    :param img:
    :param box:
    :param image_size:
    :return:
    """
    if isinstance(img, np.ndarray):
        out = cv2.resize(
            img[box[1]:box[3], box[0]:box[2]],
            (image_size, image_size),
            interpolation=cv2.INTER_AREA
        ).copy()
    else:
        out = img.crop(box).copy().resize((image_size, image_size), Image.BILINEAR)
    return out


def get_size(img):
    """
    learn from MTCNN.
    :param img:
    :return:
    """
    if isinstance(img, np.ndarray):
        return img.shape[1::-1]
    else:
        return img.size


def extract_face(img, box, image_size=160, margin=0, save_path=None):
    """
    learn from MTCNN.
    Extract face + margin from PIL Image given bounding box.

    Arguments:
        img {PIL.Image} -- A PIL Image.
        box {numpy.ndarray} -- Four-element bounding box.
        image_size {int} -- Output image size in pixels. The image will be square.
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image.
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size.
        save_path {str} -- Save path for extracted face image. (default: {None})

    Returns:
        torch.tensor -- tensor representing the extracted face.
    """
    from torchvision.transforms import functional as F
    margin = [
        margin * (box[2] - box[0]) / (image_size - margin),
        margin * (box[3] - box[1]) / (image_size - margin),
    ]
    raw_image_size = get_size(img)
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, raw_image_size[0])),
        int(min(box[3] + margin[1] / 2, raw_image_size[1])),
    ]

    face = crop_resize(img, box, image_size)

    face = F.to_tensor(np.float32(face))

    return face


def face_boxes_post_process(img, box, expand_ratio):
    """
    enlarge and crop the face patch from image
    :param img: ndarray, 1 frame from video
    :param box: output of MTCNN
    :param expand_ratio: default: 1.3
    :return:
    """
    box = [max(b, 0) for b in box]
    box_xywh = _box_mode_cvt(box)
    expand_w = int((box_xywh[2] * (expand_ratio - 1)) / 2)
    expand_h = int((box_xywh[3] * (expand_ratio - 1)) / 2)
    enlarged_box = _enlarged_bbox(box_xywh, (expand_h, expand_w))
    try:
        res = crop_bbox(img, enlarged_box)
    except ValueError:
        try:
            res = crop_bbox(img, box_xywh)
        except ValueError:
            return img
    return res


def get_face_patches(img, margin=10):
    """

    :param img: PIL Image, notice the RGB order if you open image with cv2.
    :param expand_ratio: enlarged bbox ratio, default 1.3
    :return:
    """
    face_detector = MTCNN(
        margin=margin, select_largest=False, keep_all=False, min_face_size=60,
        thresholds=[0.7, 0.8, 0.9], factor=0.709, device=device
    ).eval()
    faces, probs = face_detector(img, return_prob=True)
    return faces, probs


def get_face_patches_w_landmark(img):
    """

    :param img:
    :return:
    """
    face_detector = MTCNN(
        select_largest=False, keep_all=False, min_face_size=60,
        thresholds=[0.6, 0.8, 0.9], factor=0.709, device=device
    ).eval()
    boxes, probs, points = face_detector.detect(img, landmarks=True)
    print(boxes)
    print(probs)
    print(points)
    return boxes, probs, points


def align_landmark_crop(img, box, points):
    """

    :param img: array,
    :param box: array, [x,y,x,y]
    :param points:
    :return:
    """
    pass


if __name__ == "__main__":
    save_face_patches(frames_path="../testdata", saving_path="../testdata")
    img = cv2.imread("../dataset/frames/zzlsynxeff_009.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    get_face_patches_w_landmark(img)
