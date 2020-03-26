import os
import cv2
import pandas as pd
import json
from tqdm import tqdm


def get_video_info(video_path):
    video = cv2.VideoCapture(video_path)
    # self.ctn_format = ctn_format  # container format, eg mp4, ts...
    f_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # frame numbers in total.
    f_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # frame width
    f_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # frame height
    fps = video.get(cv2.CAP_PROP_FPS)
    codec = video.get(cv2.CAP_PROP_FOURCC)
    fmt = video.get(cv2.CAP_PROP_FORMAT)
    iso_speed = video.get(cv2.CAP_PROP_ISO_SPEED)
    hue = video.get(cv2.CAP_PROP_HUE)
    print(codec)
    print(fmt)
    return f_count, f_height, f_width, fps, codec, fmt, iso_speed, hue


video_dir = "/home/chongmin/karkin/data/dfdc_train_all"
video_info = []
for i in range(0, 50):
    if i <= 9:
        id = str(i)
    else:
        id = str(i).zfill(3)
    print(id)
    with open(os.path.join(video_dir, "dfdc_train_part_{}".format(id), "metadata.json")) as fp:
        meta_json = json.load(fp)
    for video, info in tqdm(meta_json.items()):
        label = 1 if info["label"] == "FAKE" else 0
        video_path = os.path.join(video_dir, "dfdc_train_part_{}".format(id), video)
        f_count, f_height, f_width, fps, codec, fmt, iso_speed, hue = get_video_info(video_path)
        video_info.append((video, label, f_count, f_height, f_width, fps, codec, fmt, iso_speed, hue))

video_info = pd.DataFrame(video_info, columns=["video", "label", "f_count", "f_height", "f_width", "fps", "codec", "fmt", "iso_speed", "hue"])
video_info.to_csv("../dataset/video_info.csv", index=False)
