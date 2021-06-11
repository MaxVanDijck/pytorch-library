import torch
import torch.nn as nn

class CNNBlock(nn.Module):

    def __init__(self,
                 in_channels=32,
                 out_channels=16,
                 kernel_size = 3,
                 padding=1,
                 stride=1,
                 groups=1
                 ):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(in_channels,
                             out_channels,
                             kernel_size,
                             stride,
                             padding,
                             groups=groups,
                             bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.cnn(x)
        x = self.bn(x)
        x = self.silu(x)
        return x

class SEBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SEBlock, self).__init__()
        self.pool2d = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.silu = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool2d(x)
        x = self.conv1(x)
        x = self.silu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x

class InvertedResidualBlock(nn.module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 expand_ratio,
                 reduction=4,
                 survival_prob=0.8):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)

        if self.expand: self.expand_conv = CNNBlock(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1)

        self.conv = nn.Sequential(
            CNNBlock(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim),
            SEBlock(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1 , bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def stochastic_depth(self, x):
        if not self.training:
            return x

        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, x):
        x = self.expand_conv(x) if self.expand else x

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + x
        else:
            return self.conv(x)

class EfficientNet(nn.Module):
    def __init__(self, version, num_classes):
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes)
        )

    def calculate_factors(self, version, alpha=1.2, beta=1.2):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_rate

    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        features = [CNNBlock(3, channels, 3, stride=2, padding=1)]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4*ceil(int(channels*width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock()
                )