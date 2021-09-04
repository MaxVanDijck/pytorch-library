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

class MobileNetV2(nn.Module):
    def __init__(self, 
                 num_classes=1000, 
                 width_multi=1.0, 
                 inverted_residual_setting=None, 
                 round_nearest=8, 
                 block=None, 
                 norm_layer=None):
        super(MobileNetV2, self).__init__()