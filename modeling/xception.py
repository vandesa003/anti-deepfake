"""
Xception Net.
reference: https://www.kaggle.com/greatgamedota/xception-classifier-w-ffhq-training-lb-537

Created On 23rd Feb, 2020
Author: Bohang Li
"""
from torch import nn
from pytorchcv.model_provider import get_model
from pytorchcv.models.xception import Xception


class Head(nn.Module):
    def __init__(self, in_f, out_f):
        super(Head, self).__init__()
        self.f = nn.Flatten()
        self.o = nn.Linear(in_f, out_f)

    def forward(self, x):
        x = self.f(x)
        out = self.o(x)
        return out


class BinaryXception(nn.Module):
    def __init__(self, pretrained=True):
        """

        :param in_f: default 2048.
        """
        super(BinaryXception, self).__init__()
        model = get_model("efficientnet_b0", pretrained=pretrained)
        # model = get_model("resnet18", pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])  # Remove original output layer
        self.base = model
        self.h1 = Head(1280, 1)

    def forward(self, x):
        x = self.base(x)
        return self.h1(x)
