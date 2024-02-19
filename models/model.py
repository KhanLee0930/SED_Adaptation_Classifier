import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.models import init_layer
from models.frontend import Audio_Frontend
from models.module import ConvBlock3x3, TransitionBlock, BroadcastedBlock
import whisper


class Baseline_CNN(nn.Module):
    def __init__(self, num_class=10, frontend=None):
        super(Baseline_CNN, self).__init__()
        self.conv_block1 = ConvBlock3x3(in_channels=1, out_channels=16)
        self.conv_block2 = ConvBlock3x3(in_channels=16, out_channels=32)
        self.conv_block3 = ConvBlock3x3(in_channels=32, out_channels=64)
        self.conv_block4 = ConvBlock3x3(in_channels=64, out_channels=128)

        self.fc1 = nn.Linear(128, 32, bias=True)
        self.fc_audioset = nn.Linear(32, num_class, bias=True)

        self.frontend = frontend

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input):
        # Input: (batch_size, data_length)
        if self.frontend is not None:
            x = self.frontend(input)
        # Input: (batch_size, 1, T, F)
        else:
            x = input

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.2, training=self.training)
        clipwise_output = self.fc_audioset(x)

        output_dict = {
            'clipwise_output': clipwise_output,
            'embedding': embedding}

        return output_dict


class BCResNet_Mod(torch.nn.Module):
    def __init__(self, c=4, num_class=10, frontend=None, norm=False):
        self.lamb = 0.1
        super(BCResNet_Mod, self).__init__()
        c = 10 * c
        self.c = c
        self.num_class = num_class
        self.conv1 = nn.Conv2d(1, 2 * c, 5, stride=(2, 2), padding=(2, 2))
        self.block1_1 = TransitionBlock(2 * c, c)
        self.block1_2 = BroadcastedBlock(c)

        self.block2_1 = nn.MaxPool2d(2)

        self.block3_1 = TransitionBlock(c, int(1.5 * c))
        self.block3_2 = BroadcastedBlock(int(1.5 * c))

        self.block4_1 = nn.MaxPool2d(2)

        self.block5_1 = TransitionBlock(int(1.5 * c), int(2 * c))
        self.block5_2 = BroadcastedBlock(int(2 * c))

        self.block6_1 = TransitionBlock(int(2 * c), int(2.5 * c))
        self.block6_2 = BroadcastedBlock(int(2.5 * c))
        self.block6_3 = BroadcastedBlock(int(2.5 * c))

        self.block7_1 = nn.Conv2d(int(2.5 * c), num_class, 1)

        self.block8_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.norm = norm
        if norm:
            self.one = nn.InstanceNorm2d(1)
            self.two = nn.InstanceNorm2d(int(1))
            self.three = nn.InstanceNorm2d(int(1))
            self.four = nn.InstanceNorm2d(int(1))
            self.five = nn.InstanceNorm2d(int(1))

        self.frontend = frontend

    def forward(self, input, add_noise=False, training=False, noise_lambda=0.1, k=2):

        if self.frontend is not None:
            out = self.frontend(input)
        # Input: (batch_size, 1, T, F)
        else:
            out = input

        if self.norm:
            out = self.lamb * out + self.one(out)
        out = self.conv1(out)

        out = self.block1_1(out)

        out = self.block1_2(out)
        if self.norm:
            out = self.lamb * out + self.two(out)

        out = self.block2_1(out)

        out = self.block3_1(out)
        out = self.block3_2(out)
        if self.norm:
            out = self.lamb * out + self.three(out)

        out = self.block4_1(out)

        out = self.block5_1(out)
        out = self.block5_2(out)
        if self.norm:
            out = self.lamb * out + self.four(out)

        out = self.block6_1(out)
        out = self.block6_2(out)
        out = self.block6_3(out)
        embedding = F.dropout(out, p=0.2, training=training)
        embedding = self.block8_1(embedding)
        embedding = self.block8_1(embedding)
        if self.norm:
            out = self.lamb * out + self.five(out)
        if not training and add_noise is True:
            x_hat = []
            for i in range(k):
                feat = out
                noise = (torch.rand(feat.shape) - 0.5).to('cuda') * noise_lambda * torch.std(feat)
                feat += noise
                feat = self.block7_1(feat)

                feat = self.block8_1(feat)
                feat = self.block8_1(feat)

                clipwise_output = torch.squeeze(torch.squeeze(feat, dim=2), dim=2)
                x_hat.append(clipwise_output)
            clipwise_output = x_hat

        else:
            out = self.block7_1(out)

            out = self.block8_1(out)
            out = self.block8_1(out)

            clipwise_output = torch.squeeze(torch.squeeze(out, dim=2), dim=2)

        output_dict = {
            'clipwise_output': clipwise_output,
            'embedding': embedding}

        return output_dict

    def add_new_outputs(self, n):
        block7_1 = nn.Conv2d(int(2.5 * self.c), self.num_class + n, 1)
        block7_1.weight.data[:-n] = self.block7_1.weight.data

        block7_1.to(self.block7_1.weight.device)
        self.block7_1 = block7_1
        self.num_class += n
class Audio_Encoder(nn.Module):
    def __init__(self,model,num_class):
        super().__init__()
        self.num_class = num_class
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.whisper_model = model
        self.fc1 = nn.Linear(in_features=1152000,out_features=1024,bias = True)
        self.fc2 = nn.Linear(in_features=1024,out_features=self.num_class,bias = True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, audios):
        mels = []
        for audio in audios:
            mel = whisper.log_mel_spectrogram(audio).to(self.device)
            if mel.ndim == 2: mel = mel.unsqueeze(0)
            mels.append(mel)
    # load audio and pad/trim it to fit 30 seconds
        mels = torch.cat(mels,dim=0).to(self.device)
        encoded = self.whisper_model.encoder(mels)
        encoded = encoded.view(encoded.shape[0], -1)
        print(encoded.shape)
        output = self.fc2(self.fc1(encoded))
        clipwise_output = self.softmax(output)
        output_dict = {
            'clipwise_output': clipwise_output,
            'embedding': encoded}

        return  output_dict
class AvgPool(nn.Module):
    def __init__(self):
        super(AvgPool, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=15)  # 根据输入和输出维度调整kernel_size和stride

    def forward(self, x):
        # 将输入x的形状从 (b, 1000, 256) 变为 (b, 12, 256)
        x = self.avg_pool(x.transpose(1,2)).transpose(1,2)
        return x

if __name__ == '__main__':
    panns_params = {
        'sample_rate': 48000,
        'window_size': 1024,
        'hop_size': 320,
        'mel_bins': 64,
        'fmin': 50,
        'fmax': 14000}

    frontend = Audio_Frontend(**panns_params)
    model = BCResNet_Mod(frontend=frontend)

    print(model(torch.randn(32, 48000)))

