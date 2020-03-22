import torch
from torch import nn
from pytorchcv.model_provider import get_model
from pytorchcv.models.xception import Xception
from modeling import BinaryXception, ResNet50, ResNext101


class Combine(nn.Module):
    def __init__(self):
        super(Combine, self).__init__()
        # model = get_model("xception", in_channels=3, in_size=(240, 240), pretrained=True)
        model = ResNet50(pretrained=True)
        self.cnn = nn.Sequential(*list(model.children())[:-1])
        self.rnn = nn.LSTM(2048, 128, batch_first=True, dropout=0)
        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 1)
        self.elu = nn.ELU()

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()  # batch_size, 10(frames), 3, 240, 240
        c_in = x.view(batch_size * timesteps, C, H, W)  # (10*batch, 3, 240, 240)
        c_out = self.cnn(c_in)  # (10*batch, 2048, 1, 1)
        # print(c_out.shape)
        r_in = c_out.view(batch_size, timesteps, -1)  # (batch, 10(frames), 2048)
        print(r_in.shape)
        r_out, (hn, cn) = self.rnn(r_in)  # (batch, 10(frames), 128) * 2
        print(hn.shape)
        out = self.linear1(hn)
        out = self.elu(out)
        out = self.linear2(out)
        return out.view(-1, 1)
