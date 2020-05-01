import torch
from torch import nn
from pytorchcv.model_provider import get_model
from pytorchcv.models.xception import Xception
from modeling import BinaryXception, ResNet50, ResNext101
import numpy as np


class Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(Backbone, self).__init__()
        model = get_model("efficientnet_b0", pretrained=pretrained)
        # model = get_model("resnet18", pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])  # Remove original output layer
        self.base = model
        self.fc = nn.Linear(1280, 1)

        self.init_weights()

    def init_weights(self):
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)

    def forward(self, x):
        x = self.base(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# class Combine(nn.Module):
#     def __init__(self):
#         super(Combine, self).__init__()
#         self.crnn = ConvLSTM(in_channels=3, hidden_channels=128, kernel_size=(3, 3), num_layers=1, batch_first=True)
#         self.fc = nn.Linear(128*240*240, 1)
#         self.dropout_layer = nn.Dropout(p=0.2)
#
#     def forward(self, x):
#         # print(x.shape)
#         layer_output_list, last_state_list = self.crnn(x)
#         output = self.dropout_layer(last_state_list[0][0])
#         # print(output.shape)
#         # output = output.permute(0, 2, 3, 1)
#         output = output.view(-1, 128*240*240)
#         # print(output.shape)
#         output = self.fc(output)
#         # print(output.shape)
#         output = torch.sigmoid(output)
#         return output


# class Combine(nn.Module):
#     def __init__(self):
#         super(Combine, self).__init__()
#         # model = get_model("xception", in_channels=3, in_size=(240, 240), pretrained=True)
#         model = Backbone()
#         # model = ResNet50(pretrained=True)
#         # self.cnn = nn.Sequential(*list(model.children())[:-1])
#         self.cnn = model
#         # self.bn = nn.BatchNorm2d(1280)
#         # self.rnn = nn.LSTM(1, 128, batch_first=True, dropout=0)
#         self.rnn = nn.GRU(input_size=1, hidden_size=128, batch_first=True)
#         self.linear1 = nn.Linear(128, 64)
#         self.linear2 = nn.Linear(64, 1)
#         self.init_weights()
#         self.elu = nn.ELU()
#
#     def forward(self, x):
#         batch_size, timesteps, C, H, W = x.size()  # batch_size, 10(frames), 3, 240, 240
#         c_in = x.view(batch_size * timesteps, C, H, W)  # (10*batch, 3, 240, 240)
#         # print(x.shape)
#         # c_out = []
#         # for i in range(batch_size):
#         #     c_out.append(self.cnn(x[i, :, :, :, :]).unsqueeze(0))  # (10*batch, 2048, 1, 1)
#         # print(c_out.shape)
#         c_out = self.cnn(c_in)
#         # r_in = torch.cat(c_out, dim=0)
#         r_in = c_out.view(batch_size, timesteps, -1)  # (batch, 10(frames), 2048)
#         r_in = torch.relu(r_in)
#         # print(r_in.shape)
#         # r_out, (hn, cn) = self.rnn(r_in)  # (batch, 10(frames), 128) * 2
#         r_out, hn = self.rnn(r_in)  # (batch, 10(frames), 128) * 2
#         out = self.linear1(hn)
#         out = self.elu(out)
#         out = self.linear2(out)
#         out = torch.sigmoid(out)
#         # print(out.shape)
#         return out.view(-1, 1)
#
#     def init_weights(self):
#         self.linear1.weight.data.uniform_(-0.1, 0.1)
#         self.linear1.bias.data.fill_(0)
#         self.linear2.weight.data.uniform_(-0.1, 0.1)
#         self.linear2.bias.data.fill_(0)
#         for m in self.modules():
#             if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
#                 for name, param in m.named_parameters():
#                     if 'weight_ih' in name:
#                         torch.nn.init.xavier_uniform_(param.data)
#                     elif 'weight_hh' in name:
#                         torch.nn.init.orthogonal_(param.data)
#                     elif 'bias' in name:
#                         param.data.fill_(0)


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


def conv2d_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
    return outshape


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


class ConvLSTMCell(nn.Module):
    """
    Basic CLSTM cell.
    """

    def __init__(self, in_channels, hidden_channels, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = in_channels
        self.hidden_dim = hidden_channels

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, b, h, w):
        return (torch.zeros(b, self.hidden_dim, h, w).cuda(),
                torch.zeros(b, self.hidden_dim, h, w).cuda())


class ConvLSTM(nn.Module):

    def __init__(self, in_channels, hidden_channels, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_channels = self._extend_for_multilayer(hidden_channels, num_layers)
        if not len(kernel_size) == len(hidden_channels) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = in_channels
        self.hidden_dim = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(in_channels=cur_input_dim,
                                          hidden_channels=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            b, _, _, h, w = input_tensor.shape
            hidden_state = self._init_hidden(b, h, w)

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, b, h, w):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(b, h, w))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
