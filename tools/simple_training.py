import os
import sys
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(dir_name, "../"))
import torch.nn.functional as F
from tqdm import tqdm
from modeling.xception import BinaryXception
from modeling.ResNet import ResNet50, ResNext101
from modeling.head3d import Xception3DNet, ResNet3DNet, Efficient3DNet
from modeling.CNN_LSTM import Combine
from dataloaders.dataset import PatchDataset, ConcatDataset, FinalDataset
from dataloaders.transformers import train_transformer, video_collate_fn, RandomHFlip, VideoToTensor, MinMaxNorm, pair_collate_fn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torchvision import transforms
import pandas as pd
import sklearn
from sklearn.metrics import log_loss, recall_score, precision_score
from utils.logger import init_logging
import shutil
import math
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

net = Combine().cuda()
train_data_path = "../dataset/videos/"
transformer = transforms.Compose([RandomHFlip(0.5), MinMaxNorm(), VideoToTensor()])
train_dataset = FinalDataset(
        image_folder=train_data_path,
        kind="train",
        transform=transformer
    )
trainloader = DataLoader(
        train_dataset, batch_size=2, shuffle=True,
        collate_fn=pair_collate_fn, num_workers=8, pin_memory=True
    )
criterion = nn.BCELoss(weight=None, reduction='mean')
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader, 0)):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data[0].cuda().float()
        labels = data[1].unsqueeze(1).cuda().float()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0

print('Finished Training')
