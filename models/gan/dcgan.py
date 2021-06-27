import torch
import torch.nn as nn

class DCGanDiscriminator(nn.Module):
    def __init__(self, img_channels, features):
        super(DCGanDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(img_channels, features, kernel_size=4, stride=2, padding=1)
        
        self.block1 = self._conv_block(features, features*2, 4, 2, 1)
        self.block2 = self._conv_block(features*2, features*4, 4, 2, 1)
        self.block3 = self._conv_block(features*4, features*8, 4, 2, 1)

        self.conv2 = nn.Conv2d(features*8, 1, kernel_size=4, stride=2, padding=0)

        self.leakyrelu = nn.LeakyReLU(0.2)

    def _conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.leakyrelu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.conv2(x)
        return x