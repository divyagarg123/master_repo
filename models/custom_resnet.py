import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, padding=1):
        super(ResBlock, self).__init__()
        self.in_chan = in_channel
        self.out_chan = out_channel
        self.stride = stride
        self.padding = padding
        self.res_block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_chan, out_channels=self.out_chan, kernel_size=(3, 3), stride=self.stride,
                      padding=self.padding, bias=False),
            nn.BatchNorm2d(self.out_chan),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_chan, out_channels=self.out_chan, kernel_size=(3, 3), stride=self.stride,
                      padding=self.padding, bias=False),
            nn.BatchNorm2d(self.out_chan),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.res_block(x)
        return x


class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()

        self.prep_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.R1 = ResBlock(128, 128, 1, 1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.R2 = ResBlock(512, 512, 1, 1)
        self.layer4 = nn.MaxPool2d(4)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.prep_layer(x)
        x = self.layer1(x)
        x1 = self.R1(x)
        x = x + x1
        x = self.layer2(x)
        x = self.layer3(x)
        x2 = self.R2(x)
        x = x + x2
        x = self.layer4(x)
        x = x.reshape(-1, 512)
        x = self.fc(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
