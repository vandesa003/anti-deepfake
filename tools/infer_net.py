"""
Inference for XceptionNet.

Created On 27th Feb, 2020
Author: Bohang Li
"""

import os
import sys
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(dir_name, "../"))
import torch
from modeling.xception import BinaryXception
from glob import glob
from tqdm import tqdm
from preprocess.extract_frames import FrameExtractor
from facenet_pytorch import MTCNN, InceptionResnetV1
from preprocess.face_detection import face_boxes_post_process
# TODO: infer_transformer!!!
from dataloaders.transformers import infer_transformer
from torchvision import transforms


if torch.cuda.is_available():
    device = 'cuda:0'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
print('Running on device: {0}'.format(device))


class VideoProcess(object):
    def __init__(self, video_path, face_detector, cls_model):
        fe = FrameExtractor(video_path)
        self.video_path = video_path
        self.frames = fe.fixed_frame_extract(frame_num=10)
        self.faces = []
        self.scores = []
        faces = []
        scores = []
        for idx, frame in enumerate(self.frames):
            boxes, confidences = face_detector.detect(frame)
            if boxes is None:
                continue
            else:
                best_confidence = confidences[0]
                best_box = boxes[0, :]
                best_face = face_boxes_post_process(frame, best_box, expand_ratio=1)
                faces.append(best_face)
                scores.append(best_confidence)
                # cv2.imwrite("test_face_{0}.png".format(idx), best_face)
        self.faces = faces
        self.face_scores = scores

    def classifier(self, input_size):
        print("there are: {} faces".format(len(self.faces)))
        transformer = infer_transformer
        transformed_face_list = [transformer(x) for x in self.faces]
        faces_tensor = torch.stack(transformed_face_list)
        x = torch.tensor(faces_tensor, device=torch.device("cuda")).float()
        # Make a prediction, then take the average.
        with torch.no_grad():
            y_pred = cls_model(x)
            y_pred = torch.sigmoid(y_pred.squeeze())
            print("the scores are: {}".format(y_pred))
            print("average score is: {}".format(y_pred[:len(self.faces)].mean().item()))
            return y_pred[:len(self.faces)].mean().item()


if __name__ == "__main__":
    import pandas as pd
    video = []
    face_detector = MTCNN(margin=0, select_largest=False, keep_all=False,
                          thresholds=[0.2, 0.2, 0.5], factor=0.8,
                          device=device, min_face_size=60).eval()
    cls_model = BinaryXception()
    cls_model.eval().cuda()
    cls_model.load_state_dict(torch.load("../saved_models/model_with_ffhq_balance_2.pth"))
    # res = parmap.map(fun, glob("../dataset/dfdc_train_part_0/*.mp4"), pm_processes=30, pm_pbar=True)
    for vid in tqdm(glob("../dataset/dfdc_train_part_0/*.mp4")):  # fsdrwikhge
        infer = VideoProcess(vid, face_detector, cls_model)
        score = infer.classifier(300)
        video.append((infer.video_path, len(infer.faces), score))
    report = pd.DataFrame(video, columns=["video", "num_face", "score"])
    report.to_csv("report.csv", index=False)
    print(report)
