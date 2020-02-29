import glob, os
import argparse
import pandas as pd
from sklearn.utils import shuffle


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trn_perc", type=float, default=90, help="percentage of val data.")
    return parser

def train_test_split(trn_perc):
    useful_col = ['filename', 'folder', 'label', 'original']
    patch_list = os.listdir('../dataset/face_patches/')
    meta_data = pd.read_csv('../dataset/meta_data/metadata', usecols = useful_col)

    label_list = meta_data.label.tolist()
    binary_list = []
    for label in label_list:
        if label == "FAKE":
            binary_label = 1
        elif label == "REAL":
            binary_label = 0
        binary_list.append(binary_label)
    meta_data['label'] = binary_list

    meta_data['filename_wt_type'] = meta_data['filename'].str.split('.').str[0]

    # Todo: currently remove folder 15 due to processing error
    oof_folder = 'dfdc_train_part_15'
    used_df = meta_data[meta_data.folder != oof_folder] 
    used_df = shuffle(used_df, random_state=0).reset_index(drop = True)

    trn_num = int(len(used_df)*trn_perc/100)
    trn_df = used_df[:trn_num]
    val_df = used_df[trn_num:]

    patch_df = pd.DataFrame({'patchName': patch_list})
    patch_df['filename_wt_type'] = patch_df.patchName.str.split('_').str[0]

    trn_patch = trn_df[['filename_wt_type', 'label']].merge(patch_df, how='left', on = 'filename_wt_type')
    val_patch = val_df[['filename_wt_type', 'label']].merge(patch_df, how='left', on = 'filename_wt_type')

    trn_patch = shuffle(trn_patch[['patchName', 'label']].dropna(), random_state = 0)
    val_patch = shuffle(val_patch[['patchName', 'label']].dropna(), random_state = 0)

    print('The shape for train_df and val_df are %s and %s' %(trn_patch.shape, val_patch.shape))
    trn_patch.to_csv('../dataset/trn_face_patches.csv', index = False)
    val_patch.to_csv('../dataset/val_face_patches.csv', index = False)
    

if __name__ == "__main__":
    # Get the total length of face_patches
    total_patches = len(os.listdir('../dataset/face_patches/'))
    args = get_parser().parse_args()
    trn_perc = args.trn_perc
    train_test_split(trn_perc)
    
    
    


