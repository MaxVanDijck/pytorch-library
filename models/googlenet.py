import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=out_channels, 
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
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
                                    kernel_size=3, padding=1)

        #5x5 Convolution
        self.conv_5x5_1 = nn.Conv2d(in_channels=in_channels,
                                    out_channels=filters_5x5_reduce,
                                    kernel_size=1)
        self.conv_5x5_2 = nn.Conv2d(in_channels=filters_5x5_reduce,
                                    out_channels=filters_5x5,
                                    kernel_size=5, padding=2)
        
        #Maxpool
        self.maxpool = nn.MaxPool2d(3, stride=1, padding=1)
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

class AuxiliaryBlock(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AuxiliaryBlock, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = ConvBlock(in_channels=in_channels, 
                              out_channels=128, 
                              kernel_size=1, 
                              stride=1)
        self.fc1 = nn.Linear(128, 1024)
        self.dropout = nn.Dropout(p=0.7)
        self.fc2 = nn.Linear(1024, num_classes)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Googlenet(nn.Module):
    def __init__(self, img_channels, num_classes):
        super(Googlenet, self).__init__()
        self.conv1 = ConvBlock(in_channels=img_channels,
                               out_channels=64,
                               kernel_size=7,
                               stride=2, padding=3)
        self.conv2 = ConvBlock(in_channels=64, 
                               out_channels=64, 
                               kernel_size=1, 
                               stride=1, padding=1)
        self.conv3 = ConvBlock(in_channels=64, 
                               out_channels=192, 
                               kernel_size=3, 
                               stride=1, padding=1)
        self.inception3a = InceptionBlock(in_channels=192, 
                                          filters_1x1=64, 
                                          filters_3x3_reduce=96, 
                                          filters_3x3=128, 
                                          filters_5x5_reduce=16, 
                                          filters_5x5=32, 
                                          filters_max_pool=32)
        self.inception3b = InceptionBlock(in_channels=256,
                                          filters_1x1=128,
                                          filters_3x3_reduce=128,
                                          filters_3x3=192,
                                          filters_5x5_reduce=32,
                                          filters_5x5=96,
                                          filters_max_pool=64)  
        self.inception4a = InceptionBlock(in_channels=480,
                                          filters_1x1=192,
                                          filters_3x3_reduce=96,
                                          filters_3x3=208,
                                          filters_5x5_reduce=16,
                                          filters_5x5=48,
                                          filters_max_pool=64)  
        self.inception4b = InceptionBlock(in_channels=512,
                                          filters_1x1=160,
                                          filters_3x3_reduce=112,
                                          filters_3x3=224,
                                          filters_5x5_reduce=24,
                                          filters_5x5=64,
                                          filters_max_pool=64)  
        self.inception4c = InceptionBlock(in_channels=512,
                                          filters_1x1=128,
                                          filters_3x3_reduce=128,
                                          filters_3x3=256,
                                          filters_5x5_reduce=24,
                                          filters_5x5=64,
                                          filters_max_pool=64)
        self.inception4d = InceptionBlock(in_channels=512,
                                          filters_1x1=112,
                                          filters_3x3_reduce=144,
                                          filters_3x3=288,
                                          filters_5x5_reduce=32,
                                          filters_5x5=64,
                                          filters_max_pool=64)
        self.inception4e = InceptionBlock(in_channels=528,
                                          filters_1x1=256,
                                          filters_3x3_reduce=160,
                                          filters_3x3=320,
                                          filters_5x5_reduce=32,
                                          filters_5x5=128,
                                          filters_max_pool=128)
        self.inception5a = InceptionBlock(in_channels=832,
                                          filters_1x1=256,
                                          filters_3x3_reduce=160,
                                          filters_3x3=320,
                                          filters_5x5_reduce=32,
                                          filters_5x5=128,
                                          filters_max_pool=128)    
        self.inception5b = InceptionBlock(in_channels=832,
                                          filters_1x1=384,
                                          filters_3x3_reduce=192,
                                          filters_3x3=384,
                                          filters_5x5_reduce=48,
                                          filters_5x5=128,
                                          filters_max_pool=128) 

        self.averagepool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(in_features=1024, out_features=num_classes)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.lrn = nn.LocalResponseNorm(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool2(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool2(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.averagepool(x)
        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

def GoogLeNet(img_channels=3, num_classes=1000): return Googlenet(img_channels=img_channels, num_classes=num_classes)