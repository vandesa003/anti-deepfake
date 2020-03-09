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
    file_df.to_csv(os.path.join(saving_path, 'meta_df.csv'), index=False)
    return file_df


def _check_name(name: str) -> str:
    if "_" in name:
        base = name.split("_")[0] + ".mp4"
    elif ".mp4" in name:
        base = name
    else:
        base = name
    return base


def get_full_label(file_df, basename: str):
    basename = _check_name(basename)
    label = file_df.loc[file_df['filename'] == basename, ]
    label_value = label["label"].values[0]
    if len(label) == 0:
        raise ValueError("No such video in the label file!")
    if label_value == "FAKE":
        binary_label = 1
    elif label_value == "REAL":
        binary_label = 0
    else:
        raise ValueError("label in the label file is wrong!")
    original_video = label["original"].values[0]
    folder_name = label["foldername"].values[0]
    return {"label": binary_label, "original": original_video, "foldername": folder_name}


def generate_label_csv(file_path: str, file_df, saving_path: str):
    """
    Generate the meta file based on frame level
    :param file_path:
    :param label_dict:
    :param saving_path:
    :return:
    """
    framename = []
    basename = []
    
    for file_name in tqdm(glob(os.path.join(file_path, "*.jpg"))):
        patch = os.path.basename(file_name)
        base = _check_name(patch)
        framename.append(patch)
        basename.append(base)
    res_df = pd.DataFrame({'framename': framename, 'filename': basename})
    res_df = res_df.merge(file_df, how = 'left', on = 'filename')
    res_df.to_csv(os.path.join(saving_path, "frames.csv"), index=False)
    print('The shape of generated label file is: ', res_df.shape)
    print('The shape of empty info is: ', sum(res_df.label.isna()))
    return res_df


if __name__ == "__main__":
    file_df = generate_meta_csv('../dataset/meta_data/meta_df/', '../dataset/meta_data/')
    res_df = generate_label_csv('../dataset/frames/', file_df, '../dataset/meta_data/')

