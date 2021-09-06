import torch
import torch.nn as nn

class SeperableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(SeperableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding, bias=False, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,  padding=0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DoubleSeperableBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleSeperableBlock, self).__init__()
        self.conv1 = SeperableConv(in_channels=in_channels, out_channels=out_channels)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = SeperableConv(in_channels=out_channels, out_channels=out_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

        self.residualConv = nn.Conv2d(in_channels=in_channels, 
                                      out_channels=out_channels, 
                                      kernel_size=1, 
                                      stride=2,
                                      bias=False)

    def forward(self, x):
        identity = self.residualConv(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool(x)

        x += identity
        x = self.relu(x)
        return x

class TripleSeperableBlock(nn.Module):
    def __init__(self, channels):
        super(TripleSeperableBlock, self).__init__()
        self.conv = SeperableConv(in_channels=channels, out_channels=channels, padding=1)
        self.norm = nn.BatchNorm2d(num_features=channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        for i in range(3):
            x = self.conv(x)
            x = self.norm(x)
            x = self.relu(x)

        x += identity
        return x

class Xceptionnet(nn.Module):
    def __init__(self, num_classes=1000, img_channels=3):
        super(Xceptionnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(num_features=64)

        features = []

        blockChannels = [(64, 128), (128, 256), (256, 728)]
        for i in range(3):
            in_channels, out_channels = blockChannels[i]
            features.append(DoubleSeperableBlock(in_channels=in_channels, out_channels=out_channels))

        for i in range(8):
            features.append(TripleSeperableBlock(channels=728))

        self.features = nn.Sequential(*features)

        self.conv3 = SeperableConv(in_channels=728, out_channels=728)
        self.bn3 = nn.BatchNorm2d(num_features=728)
        self.conv4 = SeperableConv(in_channels=728, out_channels=1024)
        self.bn4 = nn.BatchNorm2d(num_features=1024)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.residualConv = nn.Conv2d(in_channels=728, out_channels=1024, kernel_size=1, stride=2, bias=False)

        self.conv5 = SeperableConv(in_channels=1024, out_channels=1536)
        self.bn5 = nn.BatchNorm2d(num_features=1536)
        self.conv6 = SeperableConv(in_channels=1536, out_channels=2048)
        self.bn6 = nn.BatchNorm2d(num_features=2048)
        self.pool2 = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(in_features=2048, out_features=num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.features(x)

        identity = self.residualConv(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool1(x)

        x += identity
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        print(x.size())
        x = self.pool2(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        return x

model = Xceptionnet()
x = model(torch.randn(1, 3, 299, 299))
print(x.size())