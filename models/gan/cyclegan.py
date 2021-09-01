'''CycleGAN model for PyTorch
[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593v7)
'''

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
            layers.append(DiscBlock(in_channels, feature, stride=1 if feature==features[-1] else 2))
            in_channels = feature
        
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leakyrelu(x)
        x = self.model(x)
        return x

class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super(GenBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class GenResBlock(nn.Module):
    def __init__(self, channels):
        super(GenResBlock, self).__init__()
        self.conv1 = GenBlock(channels, channels, kernel_size=3, padding=1)
        self.conv2 = GenBlock(channels, channels, use_act=False, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        return x

class CycleGenerator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super(CycleGenerator, self).__init__()
        self.conv1 = nn.Conv2d(img_channels, 
                               num_features, 
                               kernel_size=7, 
                               stride=1, 
                               padding=3, 
                               padding_mode='reflect')
        self.norm = nn.InstanceNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = GenBlock(num_features, num_features*2, kernel_size=3, stride=2, padding=1)
        self.conv3 = GenBlock(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1)

        res_layers = [GenResBlock(num_features*4) for i in range(num_residuals)]
        self.res_block = nn.Sequential(*res_layers)

        self.conv4 = GenBlock(num_features*4, 
                              num_features*2, 
                              down=False, 
                              kernel_size=3, 
                              stride=2, 
                              padding=1, 
                              output_padding=1)
        self.conv5 = GenBlock(num_features*2, 
                              num_features*1, 
                              down=False, 
                              kernel_size=3, 
                              stride=2, 
                              padding=1, 
                              output_padding=1)
        self.conv6 = nn.Conv2d(num_features*1, 
                               img_channels, 
                               kernel_size=7, 
                               stride=1, 
                               padding=3,
                               padding_mode='reflect')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res_block(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x

def CycleGanDiscriminator(in_channels=3): return CycleDiscriminator(in_channels=in_channels)
def CycleGanGenerator(in_channels=3): return CycleGenerator(img_channels=in_channels)