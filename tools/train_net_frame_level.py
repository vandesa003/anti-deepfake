"""
Train loop for XceptionNet.

Created On 23rd Feb, 2020
Author: Bohang Li
"""
import os
import sys
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(dir_name, "../"))
import torch.nn.functional as F
from tqdm import tqdm
from modeling.xception import BinaryXception
from torch.utils.data import Dataset, DataLoader
from skimage import io as sio
from skimage import transform as tsf
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torchvision import transforms
import pandas as pd
import sklearn
from sklearn.metrics import log_loss, recall_score
from utils.logger import init_logging
import shutil


# TODO: Need to finish after data is ready!
class PatchDataset(Dataset):
    def __init__(self, image_name_list, img_folder, label_list=None, transform=None):
        self.image_name_list = image_name_list
        self.label_name_list = label_list
        self.img_folder = img_folder
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = sio.imread(os.path.join(self.img_folder, self.image_name_list[idx]))
        if self.transform:
            image = self.transform(image)
        if self.label_name_list is not None:
            label = self.label_name_list[idx]
        else:
            label = 0
        sample = {'image': image, 'label': label}
        return sample


class Rescale(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, image):
        img = tsf.resize(image, (self.height, self.width), mode='constant')
        return img


class ToTensor(object):
    """
    convert narrays in sample to Tensors.
    """
    def __call__(self, image):
        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
        # Convert image from [0, 255] to [0, 1].
        image = image / np.max(image)
        if image.shape[2] == 1:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
        # change the r,g,b to b,r,g from [0,255] to [0,1]
        # transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        tmpImg = tmpImg.transpose((2, 0, 1))
        return torch.from_numpy(tmpImg)


def criterion1(pred1, targets, weight=None):
    l1 = F.binary_cross_entropy(F.sigmoid(pred1), targets, weight=weight)
    return l1


def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
    f_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    torch.save(state, f_path)
    if is_best:
        best_fpath = os.path.join(best_model_dir, 'best_model.pt')
        shutil.copyfile(f_path, best_fpath)


def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']


def train_model(epoch, n_epochs, history, optimizer, logger=None):
    if logger is None:
        logger = init_logging(log_dir="../logs/", log_file="training.log", log_level="error")
    batch_size = 32
    model.cuda()
    model.train()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, mode='min',
        factor=0.7, verbose=True, min_lr=1e-5
    )
    total_loss = 0
    # Need to change here!
    train_data_path = "../dataset/face_patches/"
    label_csv = pd.read_csv("../dataset/trn_face_patches.csv")
    train_image_list = label_csv["patchName"]
    train_label_list = label_csv["label"]
    train_dataset = PatchDataset(
        train_image_list,
        train_data_path,
        train_label_list,
        # TODO: more data augmentation!
        transform=transforms.Compose([Rescale(300, 300), ToTensor()])
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    t = tqdm(train_loader)
    for i, data in enumerate(t):
        img_batch = data["image"]
        y_batch = data["label"]
        img_batch = img_batch.cuda().float()
        y_batch = y_batch.cuda().float()
        optimizer.zero_grad()

        out = model(img_batch)
        # weighted loss
        # class_weights = torch.tensor([0.8, 0.2])
        # weight = class_weights[y_batch.data.view(-1).long()].view_as(y_batch).cuda()
        loss = criterion1(out, y_batch, weight=None)

        total_loss += loss
        t.set_description(f'Epoch {epoch + 1}/{n_epochs}, LR: %6f, Loss: %.4f' % (
            optimizer.state_dict()['param_groups'][0]['lr'], total_loss / (i + 1)))
        if i % 20 == 0:
            logger.info(f'Epoch {epoch + 1}/{n_epochs}, LR: %6f, Loss: %.4f' % (
            optimizer.state_dict()['param_groups'][0]['lr'], total_loss / (i + 1)))

        if history is not None:
            history.loc[epoch + i / batch_size, 'train_loss'] = loss.data.cpu().numpy()
            history.loc[epoch + i / batch_size, 'lr'] = optimizer.state_dict()['param_groups'][0]['lr']

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step(total_loss)


def evaluate_model(epoch, scheduler=None, history=None, logger=None):
    if logger is None:
        logger = init_logging(log_dir="../logs/", log_file="training.log", log_level="error")
    model.cuda()
    model.eval()
    loss = 0
    pred = []
    real = []
    # Need to change here.
    batch_size = 32
    val_data_path = "../dataset/face_patches/"
    label_csv = pd.read_csv("../dataset/val_face_patches.csv")
    val_image_list = label_csv["patchName"]
    val_label_list = label_csv["label"]
    val_dataset = PatchDataset(
        val_image_list,
        val_data_path,
        val_label_list,
        transform=transforms.Compose([Rescale(300, 300), ToTensor()])
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    with torch.no_grad():
        for data in val_loader:
            img_batch = data["image"]
            y_batch = data["label"]
            img_batch = img_batch.cuda().float()
            y_batch = y_batch.cuda().float()

            o1 = model(img_batch)
            l1 = criterion1(o1, y_batch)
            loss += l1

            for j in o1:
                pred.append(F.sigmoid(j))
            for i in y_batch:
                real.append(i.data.cpu())

    pred = [p.data.cpu().numpy() for p in pred]
    pred2 = pred
    pred = [np.round(p) for p in pred]
    pred = np.array(pred)
    # TODO: it's only recall!
    acc = recall_score(real, pred, average='macro')

    real = [r.item() for r in real]
    pred2 = np.array(pred2).clip(0.1, 0.9)
    kaggle = log_loss(real, pred2)

    loss /= len(val_loader)

    if history is not None:
        history.loc[epoch, 'dev_loss'] = loss.cpu().numpy()

    if scheduler is not None:
        scheduler.step(loss)

    logger.info(f'Dev loss: %.4f, Acc: %.6f, Kaggle: %.6f' % (loss, acc, kaggle))

    return loss


if __name__ == "__main__":
    import gc
    logger = init_logging(log_dir="../logs/", log_file="training.log")
    # need to change it!!!
    device_ids = [0, 1, 2, 3]
    use_checkpoint = False
    from_best = True
    check_point_dir = "../saved_models/"
    model = BinaryXception()
    model = nn.DataParallel(model, device_ids=device_ids)
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )
    if use_checkpoint is True:
        if not from_best:
            checkpoint = os.path.join(check_point_dir, "checkpoint.pt")
        else:
            checkpoint = os.path.join(check_point_dir, "best_model.pt")
        model, optimizer, current_epoch = load_ckp(checkpoint, model, optimizer)
        logger.info("loaded checkpoint, start from epoch: {0}".format(current_epoch))
    else:
        current_epoch = 0
        logger.info("start from epoch: {0}".format(current_epoch))
    history = pd.DataFrame()
    history2 = pd.DataFrame()
    torch.cuda.empty_cache()
    gc.collect()

    best = 1e10
    n_epochs = 8
    batch_size = 128

    for epoch in range(current_epoch, n_epochs):
        torch.cuda.empty_cache()
        gc.collect()

        train_model(epoch, n_epochs, history=history, optimizer=optimizer, logger=logger)

        loss = evaluate_model(epoch, scheduler=None, history=history2, logger=logger)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_ckp(checkpoint, is_best=False, checkpoint_dir=check_point_dir, best_model_dir=check_point_dir)

        if loss < best:
            best = loss
            logger.info('Saving best model...')
            save_ckp(checkpoint, is_best=True, checkpoint_dir=check_point_dir, best_model_dir=check_point_dir)
            torch.save(model.state_dict(), '../saved_models/model.pth')

    history.to_csv("../saved_models/train_history.csv", index=False)
    history2.to_csv("../saved_models/est_history.csv", index=False)
