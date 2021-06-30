import torch
import torch.nn as nn

class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DiscBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, 
                              out_channels, 
                              4, 
                              stride, 
                              1, 
                              bias=True, 
                              padding_mode='reflect')

        self.norm = nn.InstanceNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.leakyrelu(x)
        return x