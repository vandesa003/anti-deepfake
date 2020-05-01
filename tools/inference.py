import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision import transforms
from torch import nn
from pytorchcv.model_provider import get_model
from tqdm import tqdm
import gc
import numpy as np


class InferDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        samples = []
        for img in os.listdir(image_folder):
            if img.endswith(".jpg") and os.path.isfile(os.path.join(image_folder, img)):
                samples.append(os.path.join(image_folder, img))
        self.samples = samples
        self.transform = transform
        self.hflip = VideoHFlip()

    def __getitem__(self, item):
        path = self.samples[item]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        hflip_img = self.hflip(img)
        if self.transform:
            img = self.transform(img)
            hflip_img = self.transform(hflip_img)
        img = img.unsqueeze(0)
        hflip_img = hflip_img.unsqueeze(0)
        return img, hflip_img

    def __len__(self):
        return len(self.samples)


def infer_collate_fn(batch):
    data = [item[0] for item in batch]
    data_hflip = [item[1] for item in batch]
    data = torch.cat(data, dim=0)  # B, Time, C, H, W
    data_hflip = torch.cat(data_hflip, dim=0)  # B, Time, C, H, W
    return [data, data_hflip]


class Combine(nn.Module):
    def __init__(self):
        super(Combine, self).__init__()
        self.cnn = ResCNNEncoder(CNN_embed_dim=300)
        # self.cnn = EncoderCNN()
        self.rnn = DecoderRNN(CNN_embed_dim=300, num_classes=1)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()  # batch_size, 10(frames), 3, 240, 240
        # x = x.view(batch_size * timesteps, C, H, W)  # (10*batch, 3, 240, 240)
        x = self.cnn(x)
        x = self.rnn(x)
        return x


class ResCNNEncoder(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        resnet = get_model("efficientnet_b0", pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(1280, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # # ResNet CNN
            # with torch.no_grad():
            x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
            x = x.view(x.size(0), -1)  # flatten output of conv
            # FC layers
            x = self.bn1(self.fc1(x))
            x = torch.relu(x)
            x = self.bn2(self.fc2(x))
            x = torch.relu(x)
            x = torch.dropout(x, p=self.drop_p, train=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=50):
        super(DecoderRNN, self).__init__()
        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.h_RNN = h_RNN  # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """
        # FC layers
        x = self.fc1(RNN_out[:, -1, :])  # choose RNN_out at the last time step
        x = torch.relu(x)
        x = torch.dropout(x, p=self.drop_p, train=self.training)
        x = self.fc2(x)
        return x


def get_birghtness(img):
    img_max = img.max()
    if img_max != 0:
        return img / img.max()
    else:
        return img


class MinMaxNorm(object):
    def __init__(self, num_frame=10, size=240):
        self.num_frame = num_frame
        self.size = size

    def __call__(self, frames):
        float_frames = np.zeros(frames.shape, dtype=np.float)
        for x in range(self.num_frame):
            float_frames[:, x * self.size:(x + 1) * self.size, :] = get_birghtness(
                frames[:, x * self.size:(x + 1) * self.size, :])
        return float_frames


class VideoToTensor(object):
    def __init__(self,  num_frame=10, size=240):
        self.num_frame = num_frame
        self.size = size

    def __call__(self, frames):
        # frames  # H, W, C
        tensor_list = []
        for x in range(self.num_frame):
            tensor_list.append(frames[:, x * self.size:(x + 1) * self.size, :])
        frames_tensor = torch.tensor(tensor_list)  # B, H, W, C
        return frames_tensor.permute(0, 3, 1, 2)  # B, C, H, W


class VideoHFlip(object):
    def __init__(self, num_frame=10, size=240):
        self.num_frame = num_frame
        self.size = size

    def __call__(self, frames):
        for x in range(self.num_frame):
            frames[:, x * self.size:(x + 1) * self.size, :] = cv2.flip(
                frames[:, x * self.size:(x + 1) * self.size, :], 1)
        return frames


if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    image_folder = ""
    transformer = transforms.Compose([MinMaxNorm(), VideoToTensor()])
    dataset = InferDataset(image_folder, transform=transformer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=infer_collate_fn)
    model = Combine().eval()
    pred1 = torch.empty(0, dtype=torch.float)
    pred2 = torch.empty(0, dtype=torch.float)
    for data in tqdm(dataloader):
        with torch.no_grad():
            img_batch = data[0]
            img_hflip_batch = data[1]
            img_batch = img_batch.cuda().float()
            img_hflip_batch = img_hflip_batch.cuda().float()
            o1 = model(img_batch)
            o2 = model(img_hflip_batch)
            o1 = torch.sigmoid(o1)
            o2 = torch.sigmoid(o2)
            pred1 = torch.cat((pred1, o1.cpu()), dim=0)
            pred2 = torch.cat((pred2, o2.cpu()), dim=0)
    pred1 = np.array(pred1)
    pred2 = np.array(pred2)
    pred = (pred1 + pred2) / 2

    filenames = [i.split("/")[-1].replace(".jpg", ".mp4") for i in dataset.samples]
    new_preds = []
    for x in pred:
        new_preds.append(x[0])
    print(sum(new_preds) / len(new_preds))

    for x, y in zip(new_preds, filenames):
        #     submission.loc[submission['filename']==y,'label_xception']=min([max([0.1,x]),0.9])
        submission.loc[submission['filename'] == y, 'label_xception'] = x
    plt.hist(submission['label_xception'])
