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
    fe = FrameExtractor(filename)
    fe.fixed_frame_extract(frame_num=60, start_frame_idx=10, end_frame_idx=10)
    fe.save(saving_path="../dataset/frames")


if __name__ == "__main__":
    if not os.path.isdir("../dataset/frames"):
        os.mkdir("../dataset/frames")
    args = get_parser().parse_args()
    path = args.video_path
    cpu_num = args.cpu_num
    with os.scandir(path) as entries:
        for subfile in entries:
            if subfile.is_file():
                 continue
            else:
                 print('start to process folder %s' %(subfile.name))
                 file_dir = os.path.join(path, subfile.name)
                 video_files = [file for file in glob(os.path.join(file_dir, "*.mp4"))]
                 res = parmap.map(fun, video_files, pm_processes=cpu_num, pm_pbar=True)
