import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bn_size*growth_rate, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size*growth_rate)
        self.conv2 = nn.Conv2d(bn_size*growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_rate)

        self.drop_rate = drop_rate

    def forward(self, x):
        x = torch.cat(x, 1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x

class DenseBlock(nn.ModuleDict):
    def __init__(self, 
                 num_layers, 
                 in_channels, 
                 bn_size, 
                 growth_rate, 
                 drop_rate):
        super(DenseBlock, self).__init__()
        self.layers = []
        for i in range(num_layers):
            layer = DenseLayer(
                in_channels + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.layers.append(layer)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class DenseNet(nn.Module):
    def __init__(self, 
                 growth_rate = 32, 
                 block_config=(6, 12, 24, 16), 
                 num_init_features=64, 
                 bn_size=4, 
                 drop_rate=0, 
                 num_classes=1000):
        super(DenseNet, self).__init__()
        #First Convolution
        self.features = [
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ]

        #Dense Layers
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                in_channels=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.append(block)
            num_features = num_features + num_layers * growth_rate
            #Transition
            if i != len(block_config) - 1:
                trans = nn.Sequential(
                    nn.BatchNorm2d(num_features),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_features, num_features//2, kernel_size=1, stride=1, bias=False),
                    nn.AvgPool2d(kernel_size=2, stride=2)
                )
                self.features.append(trans)
                num_features = num_features // 2

        #Out Layers
        self.features.append(nn.BatchNorm2d(num_features))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(num_features, num_classes)
        self.relu = nn.ReLU()

        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        x = self.features(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x

base_models = {
    "121": (32, (6, 12, 24, 16), 64),
    "161": (48, (6, 12, 36, 24), 96),
    "169": (32, (6, 12, 32, 32), 64),
    "201": (32, (6, 12, 48, 32), 64)
}

def DenseNet121(num_classes=1000): return DenseNet(*base_models["121"], num_classes=num_classes)
def DenseNet161(num_classes=1000): return DenseNet(*base_models["161"], num_classes=num_classes)
def DenseNet169(num_classes=1000): return DenseNet(*base_models["169"], num_classes=num_classes)
def DenseNet201(num_classes=1000): return DenseNet(*base_models["201"], num_classes=num_classes)