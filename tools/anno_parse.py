"""
This is for annotation parsing.

Created On 22th Feb, 2020
Author: Bohang Li
"""
import os
import json
import pandas as pd
from glob import glob
import argparse
from tqdm import tqdm


def merge_json(*args):
    if len(args) == 0:
        return None
    elif len(args) == 1:
        return args[0]
    else:
        res = {}
        for anno in args:
            res.update(anno)
        return res


def anno_parse(filename):
    """

    :param filename:
    :return:
    """
    with open(filename, "r") as f:
        anno = json.loads(f.read())
    return anno


def _check_name(name: str) -> str:
    if "_" in name:
        base = name.split("_")[0] + ".mp4"
    elif ".mp4" in name:
        base = name
    else:
        base = name
    return base


def get_bin_label(label_dict: dict, basename: str):
    basename = _check_name(basename)
    label = label_dict.get(basename, None)
    if label is None:
        raise ValueError("No such video in the label file!")
    if label["label"] == "FAKE":
        binary_label = 1
    elif label["label"] == "REAL":
        binary_label = 0
    else:
        raise ValueError("label in the label file is wrong!")
    return binary_label


def get_full_label(label_dict: dict, basename: str):
    basename = _check_name(basename)
    label = label_dict.get(basename, None)
    if label is None:
        raise ValueError("No such video in the label file!")
    if label["label"] == "FAKE":
        binary_label = 1
    elif label["label"] == "REAL":
        binary_label = 0
    else:
        raise ValueError("label in the label file is wrong!")
    original_video = label.get("original", "REAL")
    return {"label": binary_label, "original": original_video}


def generate_label_csv(file_path: str, label_dict: dict, saving_path: str):
    res_df = pd.DataFrame(columns=["PatchName", "Original", "Label"])
    patch_name = []
    original = []
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
        patch_name.append(basename)
    res_df["PatchName"] = patch_name
    res_df["Original"] = original
    res_df["Label"] = label
    res_df.to_csv(os.path.join(saving_path, "face_patch.csv"), index=False)
    return res_df


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("patch_file", type=str, help="path to patch file.")
    parser.add_argument("json_file", type=str, help="path to json file.")
    parser.add_argument("--saving_path", dest="saving_path", type=str, help="saving path of label csv.")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    anno = anno_parse(args.json_file)
    # anno1 = anno_parse("../dataset/dfdc_train_part_1/metadata.json")
    # anno = merge_json(anno, anno1)
    res_df = generate_label_csv(args.patch_file, anno, args.saving_path)
    print(res_df.groupby("Label").count())


