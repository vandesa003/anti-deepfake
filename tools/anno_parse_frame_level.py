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


if __name__ == "__main__":
    file_df = generate_meta_csv('../dataset/meta_data/meta_df/', '../dataset/meta_data/')
    print (file_df.head())
