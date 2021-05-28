import torch
import torch.nn as nn

class Block(nn.Module):

    def __init__(self, in_channels, out_channels, identity_downsample = None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels*2, 1)
        self.bn1 = nn.BatchNorm2d(out_channels*2)
        self.conv2 = nn.Conv2d(out_channels*2, out_channels*2, 3, stride=stride, padding=1, groups=32)
        self.bn2 = nn.BatchNorm2d(out_channels*2)
        self.conv3 = nn.Conv2d(out_channels*2, out_channels*4, 1)
        self.bn3 = nn.BatchNorm2d(out_channels*4)

        self.identity_downsample = identity_downsample

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x += identity
        x = self.relu(x)
        return x


class ResNext(nn.Module):

    def __init__(self, layers, image_channels, num_classes):
        super(ResNext, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        #Residual layers
        self.layer1 = self._make_layer(layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layer(self, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels,
                                                          out_channels*4,
                                                          kernel_size=1,
                                                          stride=stride),
                                                nn.BatchNorm2d(out_channels*4))
        
        layers.append(Block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels * 4

        for i in range(num_residual_blocks - 1):
            layers.append(Block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
        
def ResNext50(img_channels=3, num_classes=1000): return ResNext([3, 4, 6, 3], img_channels, num_classes)
def ResNext101(img_channels=3, num_classes=1000): return ResNext([3, 4, 23, 3], img_channels, num_classes)