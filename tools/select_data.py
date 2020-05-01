import os
import pandas as pd
import json
from collections import defaultdict, Counter
from tqdm import tqdm
import pickle
import random
import math
import numpy as np


def linear_sample(videos, frame_num=10):
    num = len(videos)
    idx = np.linspace(0, num - 1, frame_num, endpoint=True, dtype=np.int)
    return [videos[x] for x in idx]


meta_df = []
for i in range(0, 45):
    if i <= 9:
        id = str(i)
    else:
        id = str(i).zfill(3)
    print(id)
    meta_csv = pd.read_csv("../dataset/meta_data/dfdc_train_part_{}.csv".format(id))
    meta_df.append(meta_csv)
whole_csv = pd.concat(meta_df, ignore_index=True)
whole_csv = whole_csv[["filename", "split"]].values
data_dict = defaultdict(list)
for row in tqdm(whole_csv):
    if type(row[1]) == str:
        data_dict[row[1]].append(row[0])
with open("../dataset/origin_map_train.pkl", "wb") as fp:
    pickle.dump(dict(data_dict), fp)

meta_df = []
for i in range(45, 50):
    if i <= 9:
        id = str(i)
    else:
        id = str(i).zfill(3)
    print(id)
    meta_csv = pd.read_csv("../dataset/meta_data/dfdc_train_part_{}.csv".format(id))
    meta_df.append(meta_csv)
whole_csv = pd.concat(meta_df, ignore_index=True)
whole_csv = whole_csv[["filename", "split"]].values
data_dict = defaultdict(list)
for row in tqdm(whole_csv):
    if type(row[1]) == str:
        data_dict[row[1]].append(row[0])
with open("../dataset/origin_map_val.pkl", "wb") as fp:
    pickle.dump(dict(data_dict), fp)

print(whole_csv)

fake_videos = []
real_videos = []
face_patches_dir = "../dataset/face_patches/"

face_counter = Counter()
face_patches = defaultdict(list)
for item in tqdm(os.listdir(face_patches_dir)):
    if not item.endswith(".jpg"):
        print(item)
        continue
    basename = item.split("_")[0]
    face_patches[basename].append(item)
    face_counter[basename] += 1

non_face_real = []
non_face_fake = []
real_faces_final = []
fake_faces_final = []
for k, v in tqdm(data_dict.items()):
    real_base = k.split(".")[0]
    real_faces = face_patches[real_base]
    if len(real_faces) != 0:
        num_face_real = face_counter[real_base]
        if num_face_real < 10:
            non_face_real.append(real_base)
        else:
            real_faces_final += linear_sample(videos=sorted(real_faces))
    else:
        non_face_real.append(real_base)
    real_videos.append(k)
    # fakes = v
    fakes = random.sample(v, 1)
    for fake in fakes:
        fake_base = fake.split(".")[0]
        fake_faces = face_patches[fake_base]
        if len(fake_faces) != 0:
            num_face_fake = face_counter[fake_base]
            if num_face_fake < 10:
                non_face_fake.append(fake_base)
            else:
                fake_faces_final += linear_sample(videos=sorted(fake_faces))
        else:
            non_face_fake.append(fake_base)
        fake_videos.append(fake)

print(len(real_videos))
print(len(fake_videos))
print(len(non_face_real))
print(len(non_face_fake))
print(len(fake_faces))
print(len(real_faces))
with open("../dataset/real_faces_1_1.pkl", "wb") as fp:
    pickle.dump(real_faces_final, fp)
with open("../dataset/fake_faces_1_1.pkl", "wb") as fp:
    pickle.dump(fake_faces_final, fp)

