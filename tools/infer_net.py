"""
Inference for XceptionNet.

Created On 27th Feb, 2020
Author: Bohang Li
"""

import os
import sys
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(dir_name, "../"))
import torch
import cv2
import numpy as np
from modeling.xception import BinaryXception
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from preprocess.face_detection import get_face_patches
from glob import glob
import pandas as pd
from tqdm import tqdm
from preprocess.extract_frames import FrameExtractor
from facenet_pytorch import MTCNN, InceptionResnetV1
from preprocess.face_detection import face_boxes_post_process


if torch.cuda.is_available():
    device = 'cuda:0'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
print('Running on device: {0}'.format(device))


class VideoProcess(object):
    def __init__(self, video_path):
        fe = FrameExtractor(video_path)
        self.frames = fe.fixed_frame_extract(frame_num=10)
        faces = []
        scores = []
        for frame in self.frames:
            face_detector = MTCNN(margin=0, select_largest=False, keep_all=False,
                                  thresholds=[0.7, 0.8, 0.9], factor=0.709,
                                  device=device, min_face_size=60).eval()
            box, confidence = face_detector.detect(frame)
            if box is None:
                continue
            else:
                face = face_boxes_post_process(frame, box[0, :], expand_ratio=1)
                faces.append(face)
                scores.append(confidence)
        scores = np.array(scores)
        scores = np.flip(scores.argsort())
        print(len(scores))


if __name__ == "__main__":
    for vid in glob("../dataset/dfdc_train_part_0/*.mp4"):
        infer = VideoProcess(vid)
