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
from glob import glob
import parmap
import multiprocessing
from multiprocessing import Process, Pool


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("frames_path", type=str, help="path to frames.")
    # parser.add_argument("--cpu_num", dest="cpu_num", type=int, default=10, help="number of CPU.")
    return parser


def chunks(arr, m):
    import math
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


if __name__ == "__main__":
    gpu_list = [3, 4, 5, 6]
    multiprocessing.set_start_method('spawn')
    if not os.path.isdir("../dataset/face_patches"):
        os.mkdir("../dataset/face_patches")
    args = get_parser().parse_args()
    path = args.frames_path
    # cpu_num = args.cpu_num
    cpu_num = len(gpu_list)
    whole_path_list = glob(os.path.join(path, "*.jpg"))
    batch_path_list = chunks(whole_path_list, cpu_num)
    print(len(batch_path_list))
    p = Pool()
    for i in range(cpu_num):
        p.apply_async(save_face_patches, args=(batch_path_list[i], "../dataset/face_patches", 1.3, i, gpu_list[i]))
        # p = Process(target=save_face_patches, args=(arg_list, "../dataset/face_patches", 1.3, i, i))
    p.close()
    p.join()
    # parmap.map(save_face_patches, arg_list, saving_path="../dataset/face_patches", pm_processes=cpu_num, pm_pbar=True)
