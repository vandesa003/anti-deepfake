import glob, os
import argparse
import pandas as pd
import random
from sklearn.utils import shuffle

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trn_perc", type=float, default=90, help="percentage of trn data.")
    parser.add_argument("--ffhq", type=str2bool, default=True, help="whether to include ffhq data")
    parser.add_argument("--patches", type=str2bool, default=True, help="whether use face patches")
    return parser

def train_test_split(trn_perc, ffhq_bol, patches_bol):
    if patches_bol:
        ori_list = os.listdir('../dataset/face_patches/')
        ori_df = pd.read_csv('../dataset/meta_data/patches_ori.csv')
        ori_df['filedir'] = '../dataset/face_patches/'
        if ffhq_bol:
            ffhq_list = os.listdir('../dataset/ffhq_patches/') 
            ffhq_folder = ori_df.sample(n = len(ffhq_list), random_state = 2018).loc[:, 'foldername'].tolist()
            ffhq_df = pd.DataFrame({'subname': ffhq_list, 'label': 'REAL', 'foldername': ffhq_folder, 'filedir': '../dataset/ffhq_patches/'})    
            ori_df = pd.concat([ori_df, ffhq_df], axis = 0, sort = False).reset_index(drop = True)  
    else:
        ori_list = os.listdir('../dataset/frames/')
        ori_df = pd.read_csv('../dataset/meta_data/frames_ori.csv')
        ori_df['filedir'] = '../dataset/frames/'
        if ffhq_bol:
            ffhq_list = os.listdir('../dataset/ffhq_ori/')
            ffhq_folder = ori_df.sample(n = len(ffhq_list), random_state = 2018).loc[:, 'foldername'].tolist()
            ffhq_df = pd.DataFrame({'subname': ffhq_list, 'label': 'REAL', 'foldername': ffhq_folder, 'filedir': '../dataset/ffhq_ori/'})
            ori_df = pd.concat([ori_df, ffhq_df], axis = 0, sort = False).reset_index(drop = True)
  
    # First Shuffle, then sort based on folder name to increase the possibility of pair trainning
    frame_df = ori_df.copy()
    frame_df = shuffle(frame_df, random_state=2018)
    frame_real = frame_df.loc[frame_df['label'] == 'REAL', :]
    frame_fake = frame_df.loc[frame_df['label'] == 'FAKE', :]
    frame_real.loc[:, 'label'] = 0
    frame_fake.loc[:, 'label'] = 1

    # Balance Data
    frame_fake = frame_fake.sample(n=len(frame_real), random_state=2018)
    frame_df = shuffle(pd.concat([frame_real, frame_fake], axis=0), random_state=2018).reset_index(drop=True)
    frame_df = frame_df.sort_values(by = 'foldername').reset_index(drop = True)
    trn_num = int(len(frame_df) * trn_perc / 100)
    trn_df = frame_df.loc[:trn_num, ['subname', 'label', 'filedir']].copy()
    val_df = frame_df.loc[trn_num:, ['subname', 'label', 'filedir']].copy()
    print('The shape for train_df and val_df are %s and %s' % (trn_df.shape, val_df.shape))
    print('The file directory for trn is: \n', trn_df.filedir.value_counts())
    trn_df.to_csv('../dataset/trn_patches_{}_ffhq_{}.csv'.format(patches_bol, ffhq_bol), index=False)
    val_df.to_csv('../dataset/val_patches_{}_ffhq_{}.csv'.format(patches_bol, ffhq_bol), index=False)


if __name__ == "__main__":
    args = get_parser().parse_args()
    trn_perc = args.trn_perc
    ffhq_bol = args.ffhq
    patches_bol = args.patches
    print(ffhq_bol, patches_bol)
    train_test_split(trn_perc, ffhq_bol, patches_bol)





