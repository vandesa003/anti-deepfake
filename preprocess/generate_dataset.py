import pandas as pd
from tqdm import tqdm,trange

input_files = []
train_data_set = []
val_data_set = []
test_data_set = []

# Concat all json info into a file
def create_train_files(json_path, input_files):
    image_list = []
    label_list = []
    for files, num in tqdm(input_files, total = len(input_files)):
        json_path = os.join(json_path, file_name)
        tmp_df = pd.read_json(json_path)
        images = list(tmp_df.columns.values)
        labels = list(tmp_df.columns.label)
        image_list.append(images)
        label_list.append(labels)

# Basic Transformer of imags
def generate_transforms(image_size):