'''SqueezeNet model for PyTorch
[SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360v4)
'''

import torch
import torch.nn as nn

class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_planes, kernel_size=1)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.squeeze(x)
        x = self.relu(x)

        expand1x1_out = self.expand1x1(x)
        expand1x1_out = self.relu(expand1x1_out)

        expand3x3_out = self.expand3x3(x)
        expand3x3_out = self.relu(expand3x3_out)

        x = torch.cat([expand1x1_out, expand3x3_out], 1)
        return x

class Squeezenet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Squeezenet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 96, kernel_size=7, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.fire1 = Fire(96, 16, 64, 64)
        self.fire2 = Fire(128, 16, 64, 64)
        self.fire3 = Fire(128, 32, 128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.fire4 = Fire(256, 32, 128, 128)
        self.fire5 = Fire(256, 48, 192, 192)
        self.fire6 = Fire(384, 48, 192, 192)
        self.fire7 = Fire(384, 64, 256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.fire8 = Fire(512, 64, 256, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.pool4 = nn.AdaptiveAvgPool2d(1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.fire1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.pool2(x)
        x = self.fire4(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.pool3(x)
        x = self.fire8(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool4(x)
        x = x.reshape(x.shape[0], -1)

        return x

def SqueezeNet(in_channels=3, num_classes=1000): 
    return Squeezenet(in_channels=in_channels, num_classes=num_classes)