import torch
from torch import nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

class Conv1DBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Conv1DBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, dilation=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self._init_weight()

    def _init_weight(self):
        self._init_layer(self.conv1)
        self._init_layer(self.conv2)
        self._init_bn(self.bn1)
        self._init_bn(self.bn2)

    def _init_layer(self, layer):
        nn.init.kaiming_uniform_(layer.weight)
        if hasattr(layer, "bias"):
            if layer.bias is not None:
                layer.bias.data.fill_(0.0)

    def _init_bn(self, bn):
        bn.weight.data.fill_(1.)
        bn.bias.data.fill_(0.)

    def forward(self, x, pool_size):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=pool_size)

        return x

class Conv2DBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Conv2DBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self._init_weight()

    def _init_weight(self):
        self._init_layer(self.conv1)
        self._init_layer(self.conv2)
        self._init_bn(self.bn1)
        self._init_bn(self.bn2)

    def _init_layer(self, layer):
        nn.init.kaiming_uniform_(layer.weight)
        if hasattr(layer, "bias"):
            if layer.bias is not None:
                layer.bias.data.fill_(0.0)

    def _init_bn(self, bn):
        bn.weight.data.fill_(1.)
        bn.bias.data.fill_(0.)


    def forward(self, x, pool_size=(2, 2)):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=pool_size)

        return x

class OneDCNN(nn.Module):

    def __init__(self, classes_num=2):
        super(OneDCNN, self).__init__()

        self.pre_conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, stride=5, padding=5, bias=False)
        self.pre_bn0 = nn.BatchNorm1d(64)

        self.pre_block1 = Conv1DBlock(64, 64)
        self.pre_block2 = Conv1DBlock(64, 128)
        self.pre_block3 = Conv1DBlock(128, 128)

        self.fc1 = nn.Linear(128, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

    def forward(self, x):
        x = F.relu(self.pre_bn0(self.pre_conv0(x[:, None, :])))

        x = self.pre_block1(x, pool_size=4)
        x = self.pre_block2(x, pool_size=4)
        x = self.pre_block3(x, pool_size=4)

        y0 = torch.mean(x, dim=2)
        (y1, _) = torch.max(x, dim=2)
        x= y0 + y1

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(embedding))
        
        output_dict = {"output": clipwise_output, "embedding": embedding }

        return output_dict

class TwoDCNN(nn.Module):
    def __init__(self, classes_num=2):
        super(TwoDCNN, self).__init__()

        self.classes_num = classes_num

        self.spectrogram_extractor = Spectrogram(n_fft=1024, hop_length=320, win_length=1024, window="hann", center=True, pad_mode="reflect", freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=32000, n_fft=1024, n_mels=64, fmin=50.0, fmax=14000.0, ref=1.0, amin=1e-10, freeze_parameters=True)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = Conv2DBlock(in_channels=1, out_channels=64)
        self.conv_block2 = Conv2DBlock(in_channels=64, out_channels=128)
        self.conv_block3 = Conv2DBlock(in_channels=128, out_channels=256)
        self.conv_block4 = Conv2DBlock(in_channels=256, out_channels=512)
        self.conv_block5 = Conv2DBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = Conv2DBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, self.classes_num, bias=True)
        
        self._init_weight()

    def _init_weight(self):
        self._init_bn(self.bn0)
        self._init_layer(self.fc1)
        self._init_layer(self.fc_audioset)

    def _init_layer(self, layer):
        nn.init.kaiming_uniform_(layer.weight)
        if hasattr(layer, "bias"):
            if layer.bias is not None:
                layer.bias.data.fill_(0.0)

    def _init_bn(self, bn):
        bn.weight.data.fill_(1.)
        bn.bias.data.fill_(0.)

    def forward(self, x):
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1))
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x))

        embedding = F.dropout(x, p=0.5, training=self.training)

        pre_output = torch.sigmoid(self.fc_audioset(embedding))
        
        output_dict = {'output': pre_output, 'embedding': embedding}

        return output_dict

class OneDplusTwoDCNN(nn.Module):
    def __init__(self, classes_num=2):
        super(OneDplusTwoDCNN, self).__init__()

        self.pre_conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, stride=5, padding=5, bias=False)
        self.pre_bn0 = nn.BatchNorm1d(64)
        self.pre_block1 = Conv1DBlock(64, 64)
        self.pre_block2 = Conv1DBlock(64, 128)
        self.pre_block3 = Conv1DBlock(128, 128)

        self.pre_block4 = Conv2DBlock(in_channels=4, out_channels=64)

        self.spectrogram_extractor = Spectrogram(n_fft=1024, hop_length=320, win_length=1024, window="hann", center=True, pad_mode="reflect", freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=32000, n_fft=1024, n_mels=64, fmin=50.0, fmax=14000.0, ref=1.0, amin=1e-10, freeze_parameters=True)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = Conv2DBlock(in_channels=1, out_channels=64)
        self.conv_block2 = Conv2DBlock(in_channels=128, out_channels=128)
        self.conv_block3 = Conv2DBlock(in_channels=128, out_channels=256)
        self.conv_block4 = Conv2DBlock(in_channels=256, out_channels=512)
        self.conv_block5 = Conv2DBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = Conv2DBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        self._init_weight()

    def _init_weight(self):
        self._init_layer(self.pre_conv0)
        self._init_bn(self.pre_bn0)
        self._init_bn(self.bn0)
        self._init_layer(self.fc1)
        self._init_layer(self.fc_audioset)
 
    def _init_layer(self, layer):
        nn.init.kaiming_uniform_(layer.weight)
        if hasattr(layer, "bias"):
            if layer.bias is not None:
                layer.bias.data.fill_(0.0)

    def _init_bn(self, bn):
        bn.weight.data.fill_(1.)
        bn.bias.data.fill_(0.)

    def forward(self, i):
        a1 = F.relu(self.pre_bn0(self.pre_conv0(i[:, None, :])))

        a1 = self.pre_block1(a1, pool_size=4)
        a1 = self.pre_block2(a1, pool_size=4)
        a1 = self.pre_block3(a1, pool_size=4)

        a1 = a1.reshape((a1.shape[0], -1, 32, a1.shape[-1])).transpose(2, 3)

        a1 = self.pre_block4(a1, pool_size=(2, 1))


        x = self.spectrogram_extractor(i)
        x = self.logmel_extractor(x)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2))

        if x.shape[2] != a1.shape[2]:
            if x.shape[2] > a1.shape[2]:
                x = x[:, :, 1:, :]
            else:
                a1 = a1[:, :, 1:, :]


        x = torch.cat((x, a1), dim=1)

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1))
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x))

        embedding = F.dropout(x, p=0.5, training=self.training)

        clipwise_output = torch.sigmoid(self.fc_audioset(embedding))
        
        output_dict = {'output': clipwise_output, 'embedding': embedding}

        return output_dict

class OneDplusTwoDCNNAttention(nn.Module):
    def __init__(self, classes_num=2):
        super(OneDplusTwoDCNNAttention, self).__init__()

        self.pre_conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, stride=5, padding=5, bias=False)
        self.pre_bn0 = nn.BatchNorm1d(64)
        self.pre_block1 = Conv1DBlock(64, 64)
        self.pre_block2 = Conv1DBlock(64, 128)
        self.pre_block3 = Conv1DBlock(128, 128)
        self.pre_block4 = Conv2DBlock(in_channels=4, out_channels=64)

        self.spectrogram_extractor = Spectrogram(n_fft=1024, hop_length=320, win_length=1024, window="hann", center=True, pad_mode="reflect", freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=32000, n_fft=1024, n_mels=64, fmin=50.0, fmax=14000.0, ref=1.0, amin=1e-10, freeze_parameters=True)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = Conv2DBlock(in_channels=1, out_channels=64)
        self.conv_block2 = Conv2DBlock(in_channels=128, out_channels=128)
        self.conv_block3 = Conv2DBlock(in_channels=128, out_channels=256)
        self.conv_block4 = Conv2DBlock(in_channels=256, out_channels=512)
        self.conv_block5 = Conv2DBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = Conv2DBlock(in_channels=1024, out_channels=2048)

        self.attention = nn.MultiheadAttention(2048, 8, batch_first=True)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        self._init_weight()

    def _init_weight(self):
        self._init_layer(self.pre_conv0)
        self._init_bn(self.pre_bn0)
        self._init_bn(self.bn0)
        self._init_layer(self.fc1)
        self._init_layer(self.fc_audioset)
 
    def _init_layer(self, layer):
        nn.init.kaiming_uniform_(layer.weight)
        if hasattr(layer, "bias"):
            if layer.bias is not None:
                layer.bias.data.fill_(0.0)

    def _init_bn(self, bn):
        bn.weight.data.fill_(1.)
        bn.bias.data.fill_(0.)

    def forward(self, i):
        a1 = F.relu(self.pre_bn0(self.pre_conv0(i[:, None, :])))
        a1 = self.pre_block1(a1, pool_size=4)
        a1 = self.pre_block2(a1, pool_size=4)
        a1 = self.pre_block3(a1, pool_size=4)
        a1 = a1.reshape((a1.shape[0], -1, 32, a1.shape[-1])).transpose(2, 3)
        a1 = self.pre_block4(a1, pool_size=(2, 1))


        x = self.spectrogram_extractor(i)
        x = self.logmel_extractor(x)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2))


        if x.shape[2] != a1.shape[2]:
            
            if x.shape[2] > a1.shape[2]:
                x = x[:, :, 1:, :]
            else:
                a1 = a1[:, :, 1:, :]


        x = torch.cat((x, a1), dim=1)

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1))
        x = F.dropout(x, p=0.2, training=self.training)


        b = x.shape[0]
        d = x.shape[1]
        l2 = x.shape[2]
        l3 = x.shape[3]

        x = x.reshape((b, d, -1))
        x = x.permute(0, 2, 1)

        o, _ = self.attention(x, x, x)
        x = o.permute(0, 2, 1)
        x = x.reshape((b, d, l2, l3))

        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x))

        embedding = F.dropout(x, p=0.5, training=self.training)

        clipwise_output = torch.sigmoid(self.fc_audioset(embedding))

        output_dict = {"output": clipwise_output, "embedding": embedding}

        return output_dict
