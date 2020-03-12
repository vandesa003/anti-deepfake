import glob, os
import argparse
import pandas as pd
import random
from sklearn.utils import shuffle

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trn_perc", type=float, default=90, help="percentage of val data.")
    return parser

def train_test_split(trn_perc):
    patch_list = os.listdir('../dataset/frames/')
    frame_df = pd.read_csv('../dataset/meta_data/frames.csv')

    # First Shuffle, then sort based on folder name to increase the possibility of pair trainning
    frame_df = shuffle(frame_df, random_state=2018)
    frame_real = frame_df.loc[frame_df['label'] == 'REAL', :]
    frame_fake = frame_df.loc[frame_df['label'] == 'FAKE', :]
    frame_real['label'] = 0
    frame_fake['label'] = 1

    # Balance Data
    frame_fake = frame_fake.sample(n=len(frame_real), random_state=2018)
    frame_df = shuffle(pd.concat([frame_real, frame_fake], axis=0), random_state=2018).reset_index(drop=True)

    trn_num = int(len(frame_df) * trn_perc / 100)
    trn_df = frame_df.loc[:trn_num, ['framename', 'label']].copy()
    val_df = frame_df.loc[trn_num:, ['framename', 'label']].copy()
    print('The shape for train_df and val_df are %s and %s' % (trn_df.shape, val_df.shape))

    trn_df.to_csv('../dataset/trn_frames.csv', index=False)
    val_df.to_csv('../dataset/val_frames.csv', index=False)


if __name__ == "__main__":
    args = get_parser().parse_args()
    trn_perc = args.trn_perc
    train_test_split(trn_perc)





