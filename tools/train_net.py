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
from modeling.ResNet import ResNet50, ResNext101
from modeling.head3d import Xception3DNet
from dataloaders.dataset import PatchDataset, ConcatDataset
from dataloaders.transformers import train_transformer, video_collate_fn, RandomHFlip, VideoToTensor, MinMaxNorm
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


def criterion1(pred, targets, weight=None):
    bce_loss = nn.BCEWithLogitsLoss(weight=weight, reduction='mean')
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


def train_loop(model, dataloader, optimizer, epoch, n_epochs, history, logger=None, accumulation_steps=18):
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
    optimizer.zero_grad()
    for i, data in enumerate(t):
        img_batch = data["image"]
        y_batch = data["label"].unsqueeze(1)
        img_batch = img_batch.cuda().float()
        y_batch = y_batch.cuda().float()
        out = model(img_batch)
        loss = criterion1(out, y_batch, weight=None)

        total_loss += loss

        t.set_description(f'Epoch {epoch + 1}/{n_epochs}, LR: %6f, Loss: %.4f' % (
            optimizer.state_dict()['param_groups'][0]['lr'], total_loss / (i + 1)))

        loss = loss / accumulation_steps
        loss.backward()
        if i % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            logger.info(f'Epoch {epoch + 1}/{n_epochs}, LR: %6f, Loss: %.4f' % (
                optimizer.state_dict()['param_groups'][0]['lr'], total_loss / (i + 1)))

        if history is not None:
            history.loc[epoch + i / batch_size, 'train_loss'] = loss.data.cpu().numpy()
            history.loc[epoch + i / batch_size, 'lr'] = optimizer.state_dict()['param_groups'][0]['lr']

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
            l1 = criterion1(o1, y_batch)
            o1 = torch.sigmoid(o1)
            loss += l1
            real = torch.cat((real, y_batch.cpu()), dim=0)
            pred = torch.cat((pred, o1.cpu()), dim=0)

    pred2 = pred
    pred = [np.round(p) for p in pred]
    pred = np.array(pred)
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


def get_parser():
    parser = argparse.ArgumentParser("anti-deepfake")
    parser.add_argument("--model_saving_dir", dest="model_saving_dir", type=str,
                        default="../saved_models/patches_frame/")
    parser.add_argument("--log_file", dest="log_file", type=str, default="training_frame.log")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=64)
    parser.add_argument("--model", dest="model", type=str, default="Xception")
    parser.add_argument("--n_epochs", dest="n_epochs", type=int, default=30)
    parser.add_argument("--use_checkpoint", dest="use_checkpoint", type=int, default=0)
    parser.add_argument("--base_lr", dest="base_lr", type=float, default=0.02)
    parser.add_argument("--acc_steps", dest="acc_steps", type=int, default=8)
    parser.add_argument("--quick_test", dest="quick_test", type=int, default=0)
    return parser


if __name__ == "__main__":
    import gc
    args = get_parser().parse_args()
    # ------------------------------------Config Zone----------------------------------------
    check_point_dir = args.model_saving_dir  # checkpoint saving directory.
    model_saving_dir = check_point_dir
    log_file = args.log_file
    logger = init_logging(log_dir="../logs/", log_file=log_file)
    # need to change it!!!
    # device_ids =[i for i in range(0, 2)]  # for multi-GPU training.
    if args.use_checkpoint == 0:
        use_checkpoint = False
    else:
        use_checkpoint = True  # whether start from a checkpoint.
    from_best = True  # if start from a checkpoint, whether start from the best checkpoint.
    if not os.path.isdir(model_saving_dir):
        os.mkdir(model_saving_dir)
    if not os.path.isdir(check_point_dir):
        os.mkdir(check_point_dir)
    if "resnext" in args.model.lower():
        model = ResNext101()  # model architecture.
    elif "xception" in args.model.lower():
        model = BinaryXception()  # model architecture.
    elif "video" in args.model.lower():
        model = Xception3DNet()
    else:
        raise ValueError("model is not implemented yet!")
    # model = nn.DataParallel(model, device_ids=device_ids)

    # -------------------optimizer config.------------------
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.base_lr
    )
    acc_steps = args.acc_steps  # used for accumulate loss. must be divisible by number of batches.

    # ------------dataset and dataloader config.------------
    best = 1e10
    n_epochs = args.n_epochs  # number of training epochs.
    batch_size = args.batch_size  # number of batch size.
    num_workers = 0  # number of workers

    # -----------train dataset & dataloader-----------------
    train_data_path = "../dataset/video_img/train"

    # train_csv = pd.read_csv("../dataset/trn_frames.csv")
    # train_image_list = train_csv["framename"]
    # train_label_list = train_csv["label"]
    transformer = transforms.Compose([RandomHFlip(0.5), MinMaxNorm(), VideoToTensor()])
    train_dataset = ConcatDataset(
        image_folder=train_data_path,
        transform=transformer
    )
    # ---------------------for quick test-------------------
    if args.quick_test != 0:
        ratio = 0.001
        split_ratio = [int(ratio * len(train_dataset)), len(train_dataset) - int(ratio * len(train_dataset))]
        train_dataset, _ = random_split(train_dataset, lengths=split_ratio)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=video_collate_fn, num_workers=num_workers
    )

    # -------------val dataset & dataloader-----------------
    val_data_path = "../dataset/video_img/val"
    # val_csv = pd.read_csv("../dataset/trn_frames.csv")
    # val_image_list = val_csv["framename"]
    # val_label_list = val_csv["label"]
    transformer = transforms.Compose([MinMaxNorm(), VideoToTensor()])
    val_dataset = ConcatDataset(
        image_folder=val_data_path,
        transform=transformer
    )
    # ---------------------for quick test-------------------
    if args.quick_test != 0:
        split_ratio = [int(ratio * len(val_dataset)), len(val_dataset) - int(ratio * len(val_dataset))]
        val_dataset, _ = random_split(val_dataset, lengths=split_ratio)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_batches = math.ceil(len(train_dataset) / n_epochs)
    logger.info(
        """
        --------------------Config-----------------
        n_epochs={0}, batch_size={1}, acc_steps={2}, 
        num_of_batches={3}, acc_steps_divisible={4},
        model_saving_dir={5}
        checkpoint_dir={6}
        logfile={7}
        """.format(n_epochs, batch_size, acc_steps, num_batches, num_batches / acc_steps,
                   model_saving_dir, check_point_dir, log_file)
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

    for epoch in range(current_epoch, n_epochs):
        torch.cuda.empty_cache()
        gc.collect()

        train_loop(model, train_dataloader, optimizer, epoch, n_epochs, history, logger=logger, accumulation_steps=acc_steps)

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
