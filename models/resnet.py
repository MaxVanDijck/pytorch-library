import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    expansion = 1

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 identity_downsample=None, 
                 stride=1):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.identity_downsample = identity_downsample
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        print(x.shape)
        x = self.conv2(x)
        x = self.bn2(x)
        print(x.shape)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x += identity
        x = self.relu(x)
        return x

class TripleConv(nn.Module):
    expansion = 4

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 identity_downsample=None, 
                 stride=1):
        super(TripleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, 1)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.identity_downsample = identity_downsample
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        #Residual layers
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
        
    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, 
                                                          out_channels*block.expansion,
                                                          kernel_size=1,
                                                          stride=stride),
                                               nn.BatchNorm2d(out_channels*block.expansion))
        
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels*block.expansion 
        
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))
            
        return nn.Sequential(*layers)

def ResNet18(img_channels=3, num_classes=1000): return ResNet(DoubleConv, [3, 8, 36, 3], img_channels, num_classes)
def ResNet34(img_channels=3, num_classes=1000): return ResNet(DoubleConv, [3, 4, 6, 3], img_channels, num_classes)
def ResNet50(img_channels=3, num_classes=1000): return ResNet(TripleConv, [3, 4, 6, 3], img_channels, num_classes)
def ResNet101(img_channels=3, num_classes=1000): return ResNet(TripleConv, [3, 4, 23, 3], img_channels, num_classes)
def ResNet152(img_channels=3, num_classes=1000): return ResNet(TripleConv, [3, 8, 36, 3], img_channels, num_classes)