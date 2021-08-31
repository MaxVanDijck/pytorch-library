'''AlexNet model for PyTorch
[ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
'''

import torch
import torch.nn as nn

class Alexnet(nn.Module):
    def __init__(self, img_channels, num_classes, dropout=0.5):
        super(Alexnet, self).__init__()
        self.conv1 = nn.Conv2d(img_channels, 
                               out_channels=96, 
                               kernel_size=11,
                               stride=4)      
        self.norm1 = nn.LocalResponseNorm(96)

        self.conv2 = nn.Conv2d(in_channels=96,
                               out_channels=256,
                               kernel_size=5,
                               stride=1)
        self.norm2 = nn.LocalResponseNorm(256)

        self.conv3 = nn.Conv2d(in_channels=256,
                               out_channels=384,
                               kernel_size=3,
                               stride=1)

        self.conv4 = nn.Conv2d(in_channels=384,
                               out_channels=384,
                               kernel_size=3,
                               stride=1)
        
        self.conv5 = nn.Conv2d(in_channels=384,
                               out_channels=256,
                               kernel_size=3,
                               stride=1)
        self.norm3 = nn.LocalResponseNorm(256)

        self.fc1 = nn.Linear(256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=dropout)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.norm3(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return(x)

def AlexNet(img_channels=3, num_classes=1000, dropout=0.5): return Alexnet(img_channels=img_channels, num_classes=num_classes, dropout=dropout)