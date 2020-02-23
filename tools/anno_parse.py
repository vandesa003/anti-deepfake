"""
This is for annotation parsing.

Created On 22th Feb, 2020
Author: Bohang Li
"""
import json
import pandas as pd


def anno_parse(filename):
    """

    :param filename:
    :return:
    """
    with open(filename, "r") as f:
        anno = json.loads(f.read())

    anno = pd.read_json(filename)
    return anno


if __name__ == "__main__":
    batch = 0
    anno = anno_parse("../dataset/dfdc_train_part_{0}/metadata.json".format(batch))
    print(anno)
