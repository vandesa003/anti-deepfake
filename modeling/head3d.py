"""
This is 3d-DNN head for spatio-temporal features extraction.

Created On 14th Mar, 2020
Author: Bohang Li
"""
import os
import torch
from torch import nn
from modeling import BinaryXception, ResNet50, ResNext101
from pytorchcv.model_provider import get_model


def _get_xception_model(path="../saved_models/model_with_ffhq_balance_2.pth"):
    model = BinaryXception()
    if os.path.isfile(path):
        model.load_state_dict(torch.load(path))
    else:
        print("start from imagenet pre-trained model.")
    return model


def _get_resnext_model(path="../saved_models/model_with_ffhq_balance_2.pth"):
    model = ResNext101()
    if os.path.isfile(path):
        model.load_state_dict(torch.load(path))
    else:
        print("start from imagenet pre-trained model.")
    return model


class Head3D(nn.Module):
    """
    conv-3d model.
    input size: (batch, channel, frames, feat_w, feat_h)
    """
    def __init__(self, in_ch, out_ch, feat_h=300, feat_w=300, frame_num=10):
        super(Head3D, self).__init__()
        expansion = frame_num * feat_h * feat_w
        self.conv3d = nn.Conv3d(in_ch, out_ch, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1))
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(
            kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1))
        self.fc = nn.Linear(out_ch * expansion, 1)

    def forward(self, x):
        # batch_size, timesteps, C, H, W = x.size()  # batch_size, 10(frames), 1280, 1, 1
        x = self.conv3d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class HeadDNN(nn.Module):
    """
    Plain DNN, go through after embeddings.
    input size: (batch,  channel=frames, emb_w, emb_h)
    """
    def __init__(self, frame_num=10, emb_w=2048, emb_h=1):
        super(HeadDNN, self).__init__()
        self.fc1 = nn.Linear(frame_num * emb_w * emb_h, 256)
        self.bn = nn.BatchNorm1d(256)
        self.relu = nn.LeakyReLU()
        self.drop = nn.Dropout(p=0.15)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        print(x.shape)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class Efficient3DNet(nn.Module):
    def __init__(self, pretrained=True):
        super(Efficient3DNet, self).__init__()
        model = get_model("efficientnet_b0", pretrained=pretrained)
        self.model = nn.Sequential(*list(model.children())[:-1])  # Remove original output layer
        self.head = Head3D(in_ch=1280, out_ch=128, feat_h=1, feat_w=1, frame_num=10)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()  # batch_size, 10(frames), 3, 240, 240
        x = x.view(batch_size * timesteps, C, H, W)  # (10*batch, 3, 240, 240)
        x = self.model(x)
        x = x.view((-1, 1280, 10, 1, 1))
        x = self.head(x)
        return x


class Xception3DNet(nn.Module):
    def __init__(self, feat_h=1, feat_w=1, frame_num=10):
        super(Xception3DNet, self).__init__()
        model = _get_xception_model()
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.frame_num = frame_num
        self.model = nn.Sequential(*list(model.children())[:-1])  # Remove original output layer
        self.head = Head3D(in_ch=2048, out_ch=128, feat_h=feat_h, feat_w=feat_w, frame_num=frame_num)

    def forward(self, x):
        x = self.model(x)  # output 4-d
        # TODO: Check whether this is right! from 4d to 5d!
        x = x.view((-1, 2048, self.frame_num, self.feat_h, self.feat_w))
        x = self.head(x)  # input 5-d
        return x


class ResNet3DNet(nn.Module):
    def __init__(self, feat_h=1, feat_w=1, frame_num=10):
        super(ResNet3DNet, self).__init__()
        model = ResNet50()
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.frame_num = frame_num
        self.model = nn.Sequential(*list(model.children())[:-1])  # Remove original output layer
        self.head = Head3D(in_ch=2048, out_ch=128, feat_h=feat_h, feat_w=feat_w, frame_num=frame_num)

    def forward(self, x):
        x = self.model(x)  # output 4-d
        # TODO: Check whether this is right! from 4d to 5d!
        x = x.view((-1, 2048, self.frame_num, self.feat_h, self.feat_w))
        x = self.head(x)  # input 5-d
        return x


class XceptionDNN(nn.Module):
    def __init__(self, feat_h=1, feat_w=1, frame_num=10):
        super(XceptionDNN, self).__init__()
        model = _get_xception_model()
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.frame_num = frame_num
        self.model = nn.Sequential(*list(model.children())[:-1])  # Remove original output layer
        self.head = HeadDNN(frame_num=10, emb_w=2048, emb_h=1)

    def forward(self, x):
        x = self.model(x)  # output 4-d
        # TODO: Check whether this is right! from 4d to 5d!
        x = x.view((-1, 2048, self.frame_num, self.feat_h, self.feat_w))
        x = self.head(x)  # input 5-d
        return x
