"""
This is for annotation parsing.

Created On 22th Feb, 2020
Author: Bohang Li
"""
import os
import json
import pandas as pd
from glob import glob


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
        base = name.split("_")[0]
    elif ".mp4" in name:
        base = name.split(".")[0]
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
    original_video = label["original"]
    return {"label": binary_label, "original": original_video}


def generate_label_csv(file_path: str, label_dict: dict):
    res_df = pd.DataFrame(columns=["PatchName", "Original", "Label"])
    patch_name = []
    original = []
    label = []
    for file in glob(os.path.join(file_path, "*.jpg")):
        basename = os.path.basename(file)
        full_label = get_full_label(label_dict, basename)
        label.append(full_label["label"])
        original.append(full_label["original"])
        patch_name.append(basename)
    res_df["PatchName"] = patch_name
    res_df["Original"] = original
    res_df["Label"] = label
    res_df.to_csv("face_patch.csv", index=False)
    return res_df


if __name__ == "__main__":
    batch = 0
    anno = anno_parse("../dataset/dfdc_train_part_{0}/metadata.json".format(batch))
    print(anno)
