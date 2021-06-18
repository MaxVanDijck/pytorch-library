import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvBlockDepthwise(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1):
        super(ConvBlockDepthwise, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=in_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Mobilenet(nn.Module):
    def __init__(self, img_channels, num_classes, width_multi, shallow=False):
        super(Mobilenet, self).__init__()

        self.shallow = shallow

        self.conv1 = ConvBlock(3, out_channels=int(32*width_multi), kernel_size=3, stride=2)
        self.conv2 = ConvBlockDepthwise(in_channels=int(32*width_multi), out_channels=int(32*width_multi), kernel_size=3, stride=1)
        self.conv3 = ConvBlock(in_channels=int(32*width_multi), out_channels=int(64*width_multi), kernel_size=1, stride=1, padding=0)
        self.conv4 = ConvBlockDepthwise(in_channels=int(64*width_multi), out_channels=int(64*width_multi), kernel_size=3, stride=2)
        self.conv5 = ConvBlock(in_channels=(64*width_multi), out_channels=int(128*width_multi), kernel_size=1, stride=1, padding=0)
        self.conv6 = ConvBlockDepthwise(in_channels=(128*width_multi), out_channels=(128*width_multi), kernel_size=3, stride=1)
        self.conv7 = ConvBlock(in_channels=(128*width_multi), out_channels=(128*width_multi), kernel_size=1, stride=1, padding=0)
        self.conv8 = ConvBlockDepthwise(in_channels=(128*width_multi), out_channels=(128*width_multi), kernel_size=3, stride=2)
        self.conv9 = ConvBlock(in_channels=(128*width_multi), out_channels=(256*width_multi), kernel_size=1, stride=1, padding=0)
        self.conv10 = ConvBlockDepthwise(in_channels=(256*width_multi), out_channels=(256*width_multi), kernel_size=3, stride=1)
        self.conv11 = ConvBlock(in_channels=(256*width_multi), out_channels=(256*width_multi), kernel_size=1, stride=1, padding=0)
        self.conv12 = ConvBlockDepthwise(in_channels=(256*width_multi), out_channels=(256*width_multi), kernel_size=3, stride=2)
        self.conv13 = ConvBlock(in_channels=(256*width_multi), out_channels=(512*width_multi), kernel_size=1, stride=1, padding=0)

        if self.shallow == False:
            layers = []
            for i in range(5):
                layers.append(
                    ConvBlockDepthwise(in_channels=(512*width_multi), out_channels=(512*width_multi), kernel_size=3, stride=1),
                )
                layers.append(
                    ConvBlock(in_channels=(512*width_multi), out_channels=(512*width_multi), kernel_size=1, stride=1, padding=0),
                )
            self.extra_layers = nn.Sequential(*layers)

        self.conv14 = ConvBlockDepthwise(in_channels=(512*width_multi), out_channels=(512*width_multi), kernel_size=3, stride=2)
        self.conv15 = ConvBlock(in_channels=(512*width_multi), out_channels=(1024*width_multi), kernel_size=1, stride=1, padding=0)
        self.conv16 = ConvBlockDepthwise(in_channels=(1024*width_multi), out_channels=(1024*width_multi), kernel_size=3, stride=2)
        self.conv17 = ConvBlock(in_channels=(1024*width_multi), out_channels=(1024*width_multi), kernel_size=1, stride=1, padding=0)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features=int(1024*width_multi), out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)

        if self.shallow == False:
            x = self.extra_layers(x)

        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        print(x.shape)
        return x