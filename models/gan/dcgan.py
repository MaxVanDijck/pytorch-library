import torch
import torch.nn as nn

class DCGandiscriminator(nn.Module):
    def __init__(self, img_channels, features):
        super(DCGandiscriminator, self).__init__()
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

class DCGangenerator(nn.Module):
    def __init__(self, in_dimensions, img_channels, features):
        super(DCGangenerator, self).__init__()
        self.block1 = self._conv_block(in_dimensions, features*16, 4, 1, 0)
        self.block2 = self._conv_block(features*16, features*8, 4, 2, 1)
        self.block3 = self._conv_block(features*8, features*4, 4, 2, 1)
        self.block4 = self._conv_block(features*4, features*2, 4, 2, 1)
        self.conv = nn.ConvTranspose2d(features*2, img_channels, kernel_size=4, stride=2, padding=1)

        self.tanh = nn.Tanh()

    def _conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv(x)
        x = self.tanh(x)
        return x

def DCGanDiscriminator(in_channels=3, features=8): return DCGandiscriminator(img_channels=in_channels, features=features)
def DCGanGenerator(noise_dim = 100, in_channels=3, features=8): return DCGangenerator(in_dimensions=noise_dim, img_channels=in_channels, features=features)