import torch.nn as nn
import torch.nn.functional as F
import torch


class Ultimus(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Ultimus, self).__init__()
        self.in_chan = in_channel
        self.out_chan = out_channel
        self.k = nn.Linear(self.in_chan, self.out_chan)
        self.q = nn.Linear(self.in_chan, self.out_chan)
        self.v = nn.Linear(self.in_chan, self.out_chan)
        self.out = nn.Linear(self.out_chan, self.in_chan)

    def forward(self, x):
        k = self.k(x)
        q = self.q(x)
        v = self.v(x)
        ml = torch.matmul(torch.t(q), k)
        am = F.softmax(ml / torch.pow(torch.tensor(k.shape[-1]), 0.5), dim=-1)
        z = torch.matmul(v, am)
        out = self.out(z)
        return out


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.gap = nn.AvgPool2d(32)
        self.U1 = Ultimus(48, 8)
        self.U2 = Ultimus(48, 8)
        self.U3 = Ultimus(48, 8)
        self.U4 = Ultimus(48, 8)
        self.fc1 = nn.Linear(48, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = x.view(-1, 48)
        x1 = self.U1(x)
        x = x1 + x
        x2 = self.U2(x)
        x = x2 + x
        x3 = self.U3(x)
        x = x3 + x
        x4 = self.U4(x)
        x = x4 + x
        x = self.fc1(x)
        return F.log_softmax(x, dim=-1)
