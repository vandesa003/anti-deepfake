import os
import tqdm
import json
import glob
import pandas as pd

path = '/home/chongmin/karkin/data/dfdc_train_all/'

folder_dir = []

with os.scandir(path) as entries:
	for subfile in entries:
		if subfile.is_file():
			continue
		else:
			folder_dir.append(subfile.name)

folder_dir.sort()

# Get the json files and savaed to metadata
def create_meta_df():
	for folder in folder_dir:
		print ('Get meta info for', folder)
		folder_path = os.path.join(path, folder)
		print(folder_path)
		meta_df = pd.read_json(folder_path+'/metadata.json')
		save_dir = '../dataset/meta_data/'
		save_name = save_dir + str(folder)
		meta_df = meta_df.transpose().reset_index(drop=False)
		meta_df.columns = ['filename', 'label', 'original', 'split']
		meta_df.to_csv(save_name + '.csv', index=False)


def split_into_sub_folder():
	frame_path = '../dataset/frames/'
	meta_path = '../dataset/meta_data/meta_df/'
	for folder in folder_dir:
		print('start to copy folder', folder)
		save_dir = os.path.join(frame_path, folder)
		if not os.path.isdir(save_dir):
			os.mkdir(save_dir)

		file_df = pd.read_csv(meta_path + str(folder) + '.csv')
		file_list = file_df['filename'].tolist()
		for file_name in tqdm.tqdm(file_list):
			file_name = file_name.split('.')[0]
			file_dir = frame_path + file_name + '*'
			for file in glob.glob(file_dir):
				shutil.copy(file_dir, save_dir)

create_meta_df()
split_into_sub_folder()
	
	
