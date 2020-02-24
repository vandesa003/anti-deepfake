"""
Xception Net.
reference: https://www.kaggle.com/greatgamedota/xception-classifier-w-ffhq-training-lb-537

Created On 23rd Feb, 2020
Author: Bohang Li
"""
from torch import nn
from pytorchcv.model_provider import get_model


class Head(nn.Module):
    def __init__(self, in_f, out_f):
        super(Head, self).__init__()

        self.f = nn.Flatten()
        self.l = nn.Linear(in_f, 512)
        self.d = nn.Dropout(0.75)
        self.o = nn.Linear(512, out_f)
        self.b1 = nn.BatchNorm1d(in_f)
        self.b2 = nn.BatchNorm1d(512)
        self.r = nn.ReLU()

    def forward(self, x):
        x = self.f(x)
        x = self.b1(x)
        x = self.d(x)

        x = self.l(x)
        x = self.r(x)
        x = self.b2(x)
        x = self.d(x)

        out = self.o(x)
        return out


class BinaryXception(nn.Module):
    def __init__(self, in_f):
        """

        :param in_f: default 2048.
        """
        super(BinaryXception, self).__init__()
        model = get_model("xception", pretrained=True)
        # model = get_model("resnet18", pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])  # Remove original output layer
        self.base = model
        self.h1 = Head(in_f, 1)

    def forward(self, x):
        x = self.base(x)
        return self.h1(x)
