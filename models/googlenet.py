import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=out_channels, 
                               kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InceptionBlock(nn.Module):
    def __init__(self,
                 in_channels, 
                 filters_1x1,
                 filters_3x3_reduce,
                 filters_3x3,
                 filters_5x5_reduce,
                 filters_5x5,
                 filters_max_pool):
        super(InceptionBlock, self).__init__()
        #1x1 Convolution
        self.conv_1x1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=filters_1x1, 
                                 kernel_size=1)

        #3x3 Convolution
        self.conv_3x3_1 = nn.Conv2d(in_channels=in_channels,
                                    out_channels=filters_3x3_reduce,
                                    kernel_size=1)
        self.conv_3x3_2 = nn.Conv2d(in_channels=filters_3x3_reduce,
                                    out_channels=filters_3x3,
                                    kernel_size=3)

        #5x5 Convolution
        self.conv_5x5_1 = nn.Conv2d(in_channels=in_channels,
                                    out_channels=filters_5x5_reduce,
                                    kernel_size=1)
        self.conv_5x5_2 = nn.Conv2d(in_channels=filters_5x5_reduce,
                                    out_channels=filters_5x5,
                                    kernel_size=5)
        
        #Maxpool
        self.maxpool = nn.MaxPool2d(3)
        self.conv_maxpool = nn.Conv2d(in_channels=in_channels,
                                      out_channels=filters_max_pool,
                                      kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        #1x1 Convolution
        output_1x1 = self.conv_1x1(x)
        output_1x1 = self.relu(output_1x1)

        #3x3 Convolution
        output_3x3 = self.conv_3x3_1(x)
        output_3x3 = self.relu(output_3x3)
        output_3x3 = self.conv_3x3_2(output_3x3)
        output_3x3 = self.relu(output_3x3)

        #5x5 Convolution
        output_5x5 = self.conv_5x5_1(x)
        output_5x5 = self.relu(output_5x5)
        output_5x5 = self.conv_5x5_2(output_5x5)
        output_5x5 = self.relu(output_5x5)

        #Maxpool
        output_maxpool = self.maxpool(x)
        output_maxpool = self.conv_maxpool(output_maxpool)
        output_maxpool = self.relu(output_maxpool)

        x = torch.cat((output_1x1, output_3x3, output_5x5, output_maxpool), 1)
        return x