from torchvision import models
from torch import nn


class ResNext101(nn.Module):
    def __init__(self):
        super(ResNext101, self).__init__()
        model = models.resnext101_32x8d(pretrained=True)
        # model = nn.Sequential(*list(model.children())[:-1])  # Remove original output layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
        self.base = model

    def forward(self, x):
        return self.base(x)


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50).__init__()
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
        self.base = model
