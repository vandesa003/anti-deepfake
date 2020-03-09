"""
This is for annotation parsing for frame training.

Created On 9th March, 2020
Author: Xiaoyu Yang
"""

import os
import json
import pandas as pd
from glob import glob
import argparse
from tqdm import tqdm

def generate_meta_csv(file_path: str, saving_path: str):
    """
    return:
    foldername, filename, label, original, split
    used for match with frame info
    """
    file_list = []
    for file in tqdm(glob(os.path.join(file_path, "*.csv"))):
        res_df = pd.read_csv(file)
        res_df = res_df.transpose().reset_index(drop=False)
        res_df.columns = ['filename', 'label', 'split', 'original']
        res_df['foldername'] = file.split('/')[-1].split('.')[0]
        file_list.append(res_df)
    file_df = pd.concat(file_list, axis=0).reset_index(drop=True)
    file_df.to_csv(os.path.join(saving_path, 'meta_df.csv'))
    return file_df


def _check_name(name: str) -> str:
    if "_" in name:
        base = name.split("_")[0] + ".mp4"
    elif ".mp4" in name:
        base = name
    else:
        base = name
    return base


def get_full_label(label_info, basename: str):
    basename = _check_name(basename)
    label = label_info[basename]
    if label is None:
        raise ValueError("No such video in the label file!")
    if label["label"] == "FAKE":
        binary_label = 1
    elif label["label"] == "REAL":
        binary_label = 0
    else:
        raise ValueError("label in the label file is wrong!")
    original_video = label["original"]
    return {"label": binary_label, "original": original_video}


def generate_label_csv(file_path: str, label_dict: dict, saving_path: str):
    """
    Generate the meta file based on frame level
    :param file_path:
    :param label_dict:
    :param saving_path:
    :return:
    """
    res_df = pd.DataFrame(columns=["patchname", "original", "label", 'foldername'])
    patch_name = []
    original = []
    folder = []
    label = []
    for file in tqdm(glob(os.path.join(file_path, "*.jpg"))):
        basename = os.path.basename(file)
        try:
            full_label = get_full_label(label_dict, basename)
        except ValueError:
            print("label not found!")
            continue
        label.append(full_label["label"])
        original.append(full_label["original"])
        folder.append(full_label['foldername'])
        patch_name.append(basename)
    res_df["patchname"] = patch_name
    res_df["original"] = original
    res_df["label"] = label
    res_df["foldername"] = folder
    res_df.to_csv(os.path.join(saving_path, "face_patch.csv"), index=False)
    return res_df


if __name__ == "__main__":
    # file_df = generate_meta_csv('../dataset/meta_data/meta_df/', '../dataset/meta_data/')
    file_df = pd.read_csv("../dataset/meta_data/meta_df.csv")
    res_df = generate_label_csv('../dataset/frames/', file_df, '../dataset/meta_data/')
