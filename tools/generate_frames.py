"""
tools for frame generation.

Created On 23rd Feb, 2020
Author: Bohang Li
"""
import sys
import os
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(dir_name, "../"))
from preprocess.extract_frames import FrameExtractor
import argparse
import parmap
from glob import glob


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="path to videos.")
    parser.add_argument("--cpu_num", type=int, default=2, help="number of cpu core used.")
    return parser


def fun(filename):
    fe = FrameExtractor(filename, extract_fraq=30)
    fe.save(saving_path="../dataset/frames")


if __name__ == "__main__":
    if not os.path.isdir("../dataset/frames"):
        os.mkdir("../dataset/frames")
    args = get_parser().parse_args()
    path = args.video_path
    cpu_num = args.cpu_num
    video_files = [file for file in glob(os.path.join(path, "*.mp4"))]
    res = parmap.map(fun, video_files, pm_processes=cpu_num, pm_pbar=True)
