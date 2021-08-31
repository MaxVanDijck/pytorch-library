'''VGGNet model for PyTorch
[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556v6)
'''

import torch
import torch.nn as nn

models = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu= nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Vggnet(nn.Module):
    def __init__(self, img_channels, num_classes, version):
        super(Vggnet, self).__init__()
        self.img_channels = img_channels
        self.version = version
        self.conv_layers = self.create_architecture()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
        
    def create_architecture(self):
        layers = []
        in_channels = self.img_channels

        for i in self.version:
            if type(i) == int:
                layers += [
                    ConvBlock(in_channels=in_channels,
                                 out_channels = i),
                ]
                in_channels = i
            elif i == 'M':
                layers += [
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]  
        return nn.Sequential(*layers)

def VGGNet11(img_channels=3, num_classes=1000): 
    return Vggnet(img_channels=img_channels, num_classes=num_classes, version=models['11'])
def VGGNet13(img_channels=3, num_classes=1000): 
    return Vggnet(img_channels=img_channels, num_classes=num_classes, version=models['13'])
def VGGNet16(img_channels=3, num_classes=1000): 
    return Vggnet(img_channels=img_channels, num_classes=num_classes, version=models['16'])
def VGGNet19(img_channels=3, num_classes=1000): 
    return Vggnet(img_channels=img_channels, num_classes=num_classes, version=models['19'])