'''MobileNetV2 model for PyTorch
[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381v4)
'''

import torch
import torch.nn as nn

def _make_divisible(value, divisor, minValue=None):
    if minValue is None:
        minValue = divisor
    newValue = max(minValue, int(value + divisor/2) // divisor * divisor)
    if newValue < 0.9 * value:
        newValue += divisor
    return newValue

class ConvLayer(nn.Module):
    def __init__(self, in_channels, 
                out_channels, 
                kernel_size=3, 
                stride=1, 
                groups=1, 
                norm_layer=None):
        super(ConvLayer, self).__init__()
        padding = (kernel_size-1) // 2
        layers = []
        layers.append(
            nn.Conv2d(in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=kernel_size, 
                    stride=stride, 
                    padding=padding,
                    groups=groups, 
                    bias=False)
        )
        if norm_layer is None:
            layers.append(nn.BatchNorm2d(out_channels))
        else:
            layers.append(norm_layer(out_channels))

        layers.append(nn.ReLU6(inplace=True))
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        return x

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, norm_layer=None):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2], "Stride not equal to 1 or 2"

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_residual = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(ConvLayer(in_channels, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.append(ConvLayer(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer))
        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False))

        if norm_layer is None:
            layers.append(nn.BatchNorm2d(out_channels))
        else:
            layers.append(norm_layer(out_channels))

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.features(x)
        else:
            return self.features(x)

class Mobilenetv2(nn.Module):
    def __init__(self, 
                 num_classes=1000, 
                 width_multi=1.0, 
                 inverted_residual_setting=None, 
                 round_nearest=8, 
                 block=None, 
                 norm_layer=None):
        super(Mobilenetv2, self).__init__()
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # expansion factor, output channels, repeating number, stride
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        #Check inverted_residual_setting
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        #first layer
        input_channel = _make_divisible(input_channel * width_multi, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_multi), round_nearest)
        features = [ConvLayer(3, input_channel, stride=2, norm_layer=norm_layer)]
        #inverted residual blocks
        for t, c, n, s, in inverted_residual_setting:
            output_channel = _make_divisible(c * width_multi, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidualBlock(input_channel, 
                                      output_channel, 
                                      stride, 
                                      expand_ratio=t, 
                                      norm_layer=norm_layer))
                input_channel = output_channel

        #last layers
        features.append(ConvLayer(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))

        self.features = nn.Sequential(*features)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(self.last_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

def MobileNetV2(num_classes=1000, 
                 inverted_residual_setting=None, 
                 round_nearest=8, 
                 block=None, 
                 norm_layer=None):
    return Mobilenetv2(num_classes, 
                       inverted_residual_setting,
                       round_nearest, 
                       block, 
                       norm_layer)