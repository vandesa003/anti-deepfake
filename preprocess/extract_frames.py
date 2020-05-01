"""
extract frames from videos.

Created On 22th Feb, 2020
Author: Bohang Li
"""
import os
import cv2


class FrameExtractor(object):
    def __init__(self, video_path):
        video = cv2.VideoCapture(video_path)
        self.video_path = video_path
        self.base_name = os.path.basename(video_path).split(".")[0]
        self.video = video  # cv2.VideoCapture instance
        # self.ctn_format = ctn_format  # container format, eg mp4, ts...
        self.f_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))  # frame numbers in total.
        self.f_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))  # frame width
        self.f_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # frame height
        self.key_frames = []

    def fixed_frame_extract(self, frame_num, start_frame_idx=10, end_frame_idx=10):
        """
        extract frames by fixed frame number.
        :param frame_num: int, frame number.
        :param start_frame_idx: int, start frame index.
        :param end_frame_idx: int, end frame index.
        :return: list, a list of frames.
        """
        if frame_num > (self.f_count - start_frame_idx - end_frame_idx):
            raise ValueError("extract frame number is larger than total frame number.")
        frames = []
        interval = int((self.f_count - start_frame_idx - end_frame_idx) / frame_num) + 1
        count = 0
        ret = True
        while ret and len(frames) < frame_num:
            count += 1
            ret, frame = self.video.read()
            if count < start_frame_idx or count > (self.f_count - end_frame_idx):
                continue
            if frame is None:
                continue
            if (count-start_frame_idx) % interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert color to RGB!
                frames.append(frame)
        self.video.release()
        self.key_frames = frames
        return frames

    def fixed_time_extract(self, extract_fraq):
        """
        extract frames by fixed internal time
        :param extract_fraq: int, extract internal, for example, 5 means extract 1 frame per 5 frames.
        :return: list, a list of frames.
        """
        frames = []
        if extract_fraq > self.f_count:
            raise ValueError("extract fraq is larger than total frame number.")
        count = 0
        ret = True
        while ret:
            ret, frame = self.video.read()
            if frame is None:
                continue
            if count % extract_fraq == 0:
                frames.append(frame)
            count += 1
        self.video.release()
        self.key_frames = frames
        return frames

    def save(self, saving_path):
        for ids, frame in enumerate(self.key_frames):
            filename = os.path.join(saving_path, self.base_name + "_" + "{0}".format(str(ids)).zfill(3) + ".jpg")
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                filename=filename,
                img=bgr_frame
            )


if __name__ == "__main__":
    batch = 0
    filename = "uaspniazcl.mp4"
    fe = FrameExtractor("../dataset/dfdc_train_part_{0}/{1}".format(batch, filename))
    fe.fixed_frame_extract(frame_num=10)
    print("video: {0}, fps: {1: .2f}".format(fe.base_name, fe.f_count / 10))
    fe.save(saving_path="../testdata/")
