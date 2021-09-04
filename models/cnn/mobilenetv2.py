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

        layers.append(nn.ReLU6(inplace=True))

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        return x
            

