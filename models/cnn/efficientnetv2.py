'''EfficientNetV2 model for PyTorch
[EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298v3)
'''

import torch
import torch.nn as nn
import math

def _make_divisible(value, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor/2) // divisor * divisor)
    
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value

class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(oup, _make_divisible(inp // reduction, 8))
        self.fc2 = nn.Linear(_make_divisible(inp // reduction, 8), oup)

        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc1(y)
        y = self.silu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y

class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if use_se:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                SELayer(inp, hidden_dim),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Efficientnetv2(nn.Module):
    def __init__(self, cfgs, in_channels=3, num_classes=1000, width_multi=1.0):
        super(Efficientnetv2, self).__init__()
        self.cfgs = cfgs
        
        #First layer
        input_channels = _make_divisible(24 * width_multi, 8)
        layers = [self.conv_3x3_bn(3, input_channels, 2)]
        #Inverted Residual Blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channels = _make_divisible(c * width_multi, 8)
            for i in range(n):
                layers.append(block(input_channels, output_channels, s if i == 0 else 1, t, use_se))
                input_channels = output_channels
        self.features = nn.Sequential(*layers)
        #Out layers
        output_channels = _make_divisible(1792 * width_multi, 8) if width_multi > 1.0 else 1792
        self.conv = self.conv_1x1_bn(input_channels, output_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(output_channels, num_classes)

        self.initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def conv_3x3_bn(self, inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.SiLU()
        )

    def conv_1x1_bn(self, inp, oup):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(oup),
            nn.SiLU()
        )

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()

configs = {
        # t, c, n, s, SE
    's' : [
        [1,  24,  2, 1, 0],
        [4,  48,  4, 2, 0],
        [4,  64,  4, 2, 0],
        [4, 128,  6, 2, 1],
        [6, 160,  9, 1, 1],
        [6, 256, 15, 2, 1],
    ],
    'm' : [
        [1,  24,  3, 1, 0],
        [4,  48,  5, 2, 0],
        [4,  80,  5, 2, 0],
        [4, 160,  7, 2, 1],
        [6, 176, 14, 1, 1],
        [6, 304, 18, 2, 1],
        [6, 512,  5, 1, 1],
    ],
    'l' : [
        [1,  32,  4, 1, 0],
        [4,  64,  7, 2, 0],
        [4,  96,  7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1],
        [6, 384, 25, 2, 1],
        [6, 640,  7, 1, 1],
    ],
    'xl' : [
        [1,  32,  4, 1, 0],
        [4,  64,  8, 2, 0],
        [4,  96,  8, 2, 0],
        [4, 192, 16, 2, 1],
        [6, 256, 24, 1, 1],
        [6, 512, 32, 2, 1],
        [6, 640,  8, 1, 1],
    ]
}

def EfficientNetV2S(in_channels = 3, num_classes=1000, **kwargs):
    return Efficientnetv2(cfgs = configs['s'], in_channels=in_channels, num_classes=num_classes, **kwargs)
def EfficientNetV2M(in_channels = 3, num_classes=1000, **kwargs):
    return Efficientnetv2(cfgs = configs['m'], in_channels=in_channels, num_classes=num_classes, **kwargs)
def EfficientNetV2L(in_channels = 3, num_classes=1000, **kwargs):
    return Efficientnetv2(cfgs = configs['l'], in_channels=in_channels, num_classes=num_classes, **kwargs)
def EfficientNetV2XL(in_channels = 3, num_classes=1000, **kwargs):
    return Efficientnetv2(cfgs = configs['xl'], in_channels=in_channels, num_classes=num_classes, **kwargs)