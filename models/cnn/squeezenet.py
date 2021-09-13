import torch
import torch.nn as nn

class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_planes, kernel_size=1)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3)

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