"""
tools for frame generation.

Created On 23rd Feb, 2020
Author: Bohang Li
"""

import sys
import os
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(dir_name, "../"))
from preprocess.face_detection import save_face_patches
import argparse
import parmap
from glob import glob


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("frames_path", type=str, help="path to frames.")
    return parser


if __name__ == "__main__":
    if not os.path.isdir("../dataset/face_patches"):
        os.mkdir("../dataset/face_patches")
    args = get_parser().parse_args()
    path = args.frames_path
    save_face_patches(frames_path=path, saving_path="../dataset/face_patches")
