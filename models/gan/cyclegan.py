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

class CycleDiscriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super(CycleDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               features[0], 
                               kernel_size=4, 
                               stride=2, 
                               padding=1, 
                               padding_mode='reflect')
        self.leakyrelu = nn.LeakyReLU()

        layers=[]
        in_channels = features[0]

        for feature in features[1:]:
            layers.append(DiscBlock(in_channels, feature, stride=1 if features==features[-1] else 2))
            in_channels = feature
        
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leakyrelu(x)
        x = self.model(x)
        return x