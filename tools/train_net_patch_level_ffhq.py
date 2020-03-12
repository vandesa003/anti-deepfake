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
from dataloaders.dataset import PatchDataset, PatchDatasetFFHQ
from dataloaders.transformers import train_transformer
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


def criterion1(pred, targets, weight=None):
    # l1 = F.binary_cross_entropy(pred, targets, weight=weight)
    bce_loss = nn.BCELoss(weight=weight, reduction='mean')
    loss = bce_loss(pred, targets)
    return loss


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


def train_loop(model, dataloader, optimizer, epoch, n_epochs, history, logger=None):
    if logger is None:
        logger = init_logging(log_dir="../logs/", log_file="training.log", log_level="error")
    model.cuda()
    model.train()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=300, mode='min',
        factor=0.7, verbose=True, min_lr=1e-5
    )
    total_loss = 0
    # Need to change here!
    t = tqdm(dataloader)
    for i, data in enumerate(t):
        img_batch = data["image"]
        y_batch = data["label"].unsqueeze(1)
        img_batch = img_batch.cuda().float()
        y_batch = y_batch.cuda().float()
        optimizer.zero_grad()
        out = model(img_batch)
        loss = criterion1(torch.sigmoid(out), y_batch, weight=None)

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


def evaluate_model(model, dataloader, epoch, scheduler=None, history=None, logger=None):
    if logger is None:
        logger = init_logging(log_dir="../logs/", log_file="training.log", log_level="error")
    model.cuda()
    model.eval()
    loss = 0
    real = torch.empty(0, dtype=torch.float)
    pred = torch.empty(0, dtype=torch.float)
    with torch.no_grad():
        for data in dataloader:
            img_batch = data["image"]
            y_batch = data["label"].unsqueeze(1)
            img_batch = img_batch.cuda().float()
            y_batch = y_batch.cuda().float()

            o1 = model(img_batch)
            o1 = torch.sigmoid(o1)
            l1 = criterion1(o1, y_batch)
            loss += l1
            real = torch.cat((real, y_batch.cpu()), dim=0)
            pred = torch.cat((pred, o1.cpu()), dim=0)
    # pred = [p.data.cpu().numpy() for p in pred]
    pred2 = pred
    pred = [np.round(p) for p in pred]
    pred = np.array(pred)
    real = np.array(real)
    recall = recall_score(real, pred, average='macro')
    precision = precision_score(real, pred, average="macro")

    real = [r.item() for r in real]
    kaggle = log_loss(real, pred2, eps=1e-7)

    loss /= len(dataloader)

    if history is not None:
        history.loc[epoch, 'dev_loss'] = loss.cpu().numpy()

    if scheduler is not None:
        scheduler.step(loss)

    logger.info("evaluation: Dev loss: {0:.4f}, Recall: {1:.6f}, Precision: {2:.6f}, Kaggle: {3:.6f}"
                .format(loss, recall, precision, kaggle))

    return loss


if __name__ == "__main__":
    import gc

    # ------------------------------------Config Zone----------------------------------------
    logger = init_logging(log_dir="../logs/", log_file="training_patches_ffhq.log")
    # need to change it!!!
    # device_ids =[i for i in range(0, 2)]  # for multi-GPU training.
    use_checkpoint = False  # whether start from a checkpoint.
    from_best = True  # if start from a checkpoint, whether start from the best checkpoint.
    check_point_dir = "../saved_models/patches_ffhqs/"  # checkpoint saving directory.
    model_saving_dir = check_point_dir
    if not os.path.isdir(model_saving_dir):
        os.mkdir(model_saving_dir)

    if not os.path.isdir(check_point_dir):
        os.mkdir(check_point_dir)
    model = BinaryXception()  # model architecture.
    # model = nn.DataParallel(model, device_ids=device_ids)

    # -------------------optimizer config.------------------
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.02
    )

    # ------------dataset and dataloader config.------------
    best = 1e10
    n_epochs = 30  # number of training epochs.
    batch_size = 64  # number of batch size.
    num_workers = 3  # number of workers

    # -----------train dataset & dataloader-----------------
    train_data_path = "../dataset/face_patches/"
    train_csv = pd.read_csv("../dataset/trn_patches_True_ffhq_True.csv")
    train_image_list = train_csv["subname"]
    train_label_list = train_csv["label"]
    train_folder_list = train_csv["filedir"]
    transformer = train_transformer
    train_dataset = PatchDatasetFFHQ(
        train_image_list,
        train_folder_list,
        train_label_list,
        transform=transformer
    )
    # ---------------------for quick test-------------------
    #ratio = 0.001
    #split_ratio = [int(ratio * len(train_dataset)), len(train_dataset) - int(ratio * len(train_dataset))]
    #train_dataset, _ = random_split(train_dataset, lengths=split_ratio)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # -------------val dataset & dataloader-----------------
    val_data_path = train_data_path
    val_csv = pd.read_csv("../dataset/val_patches_True_ffhq_True.csv")
    val_image_list = val_csv["subname"]
    val_label_list = val_csv["label"]
    val_folder_list = val_csv["filedir"]
    transformer = train_transformer
    val_dataset = PatchDatasetFFHQ(
        val_image_list,
        val_folder_list,
        val_label_list,
        transform=transformer
    )
    # ---------------------for quick test-------------------
    #split_ratio = [int(ratio * len(val_dataset)), len(val_dataset) - int(ratio * len(val_dataset))]
    #val_dataset, _ = random_split(val_dataset, lengths=split_ratio)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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

    for epoch in range(current_epoch, n_epochs):
        torch.cuda.empty_cache()
        gc.collect()

        train_loop(model, train_dataloader, optimizer, epoch, n_epochs, history, logger=logger)

        loss = evaluate_model(model, val_dataloader, epoch, scheduler=None, history=history2, logger=logger)

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
            torch.save(model.state_dict(), os.path.join(model_saving_dir, 'model.pth'))

    history.to_csv(os.path.join(model_saving_dir, "train_history.csv"), index=False)
    history2.to_csv(os.path.join(model_saving_dir, "test_history.csv"), index=False)