import os
import pandas as pd
import numpy as np
import shutil

patches_file = "../dataset/video_img_all/"
train_part = [0, 40]
val_part = [40, 50]
train_filenames = []
for i in range(train_part[0], train_part[1]):
    if i <= 9:
        id = str(i)
    else:
        id = str(i).zfill(3)
    print(id)
    meta_csv = pd.read_csv("../dataset/meta_data/dfdc_train_part_{}.csv".format(id))
    train_filenames += list(meta_csv["filename"].values.squeeze())

val_filenames = []
for i in range(val_part[0], val_part[1]):
    if i <= 9:
        id = str(i)
    else:
        id = str(i).zfill(3)
    meta_csv = pd.read_csv("../dataset/meta_data/dfdc_train_part_{}.csv".format(id))
    val_filenames += list(meta_csv["filename"].values.squeeze())

print(len(train_filenames))
# for face_patch in os.listdir(patches_file):
#     print(face_patch)
#     if face_patch.endswith(".jpg"):
#         if face_patch.split("_")[0] + ".mp4" in train_filenames:
#             shutil.copyfile(os.path.join(patches_file, face_patch),
#                             os.path.join("../dataset/video_img_batch/train", face_patch))
#         elif face_patch.split("_")[0] + ".mp4" in val_filenames:
#             shutil.copyfile(os.path.join(patches_file, face_patch),
#                             os.path.join("../dataset/video_img_batch/val", face_patch))

trn_concat = pd.read_csv("trn_patches_ffhq_false_img_concat.csv")
val_concat = pd.read_csv("val_patches_ffhq_false_img_concat.csv")
whole_data = trn_concat.append(val_concat).values
print(whole_data)

img_train = []
img_train_label = []
img_val = []
img_val_label = []
for r_id in range(len(whole_data)):
    filename, label = whole_data[r_id][0], whole_data[r_id][1]
    if filename.split("_")[0] + ".mp4" in train_filenames:
        img_train.append(filename)
        img_train_label.append(label)
    elif filename.split("_")[0] + ".mp4" in val_filenames:
        img_val.append(filename)
        img_val_label.append(label)
    else:
        raise ValueError("wrong sample!")

batch_train = pd.DataFrame({"subname": img_train, "label": img_train_label})
batch_val = pd.DataFrame({"subname": img_val, "label": img_val_label})
batch_train.to_csv("../dataset/batch_train_concat_0322.csv", index=False)
batch_val.to_csv("../dataset/batch_val_concat_0322.csv", index=False)


import os
import shutil
import pandas as pd
err_csv = pd.read_csv("../logs/error_info.csv").values
dst = "../dataset/error_video"
if not os.path.isdir(dst):
    os.mkdir(dst)
for i in err_csv:
    basename = os.path.basename(i[0])
    shutil.copyfile(i, os.path.join(dst, basename))

