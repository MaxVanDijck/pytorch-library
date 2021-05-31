import torch
import torch.nn as nn

class CNNBlock(nn.Module):

    def __init__(self,
                 in_channels=32,
                 out_channels=16,
                 kernel_size = 3,
                 padding=1,
                 stride=1,
                 groups=1
                 ):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(in_channels,
                             out_channels,
                             kernel_size,
                             stride,
                             padding,
                             groups=groups,
                             bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.cnn(x)
        x = self.bn(x)
        x = self.silu(x)
        return x

class SEBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SEBlock, self).__init__()
        self.pool2d = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.silu = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool2d(x)
        x = self.conv1(x)
        x = self.silu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x