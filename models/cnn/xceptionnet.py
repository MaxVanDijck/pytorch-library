import torch
import torch.nn as nn

class SeperableBlock(nn.Module):
    def __init__(self, channels):
        super(SeperableBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, 
                               out_channels=channels, 
                               kernel_size=3, 
                               groups=channels, 
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=channels, 
                               out_channels=channels, 
                               kernel_size=3, 
                               groups=channels, 
                               bias=False)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU()

        self.residualConv = nn.Conv2d(in_channels=channels, 
                                      out_channels=channels, 
                                      kernel_size=1, 
                                      stride=2, 
                                      bias=False)

    def forward(self, x):
        identity = self.residualConv(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pool(x)

        x += identity
        x = self.relu(x)
        return x