import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, 
                 in_channels, 
                 growth_rate, 
                 bn_size, 
                 drop_rate, 
                 memory_efficient=False):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bn_size*growth_rate, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size*growth_rate)
        self.conv2 = nn.Conv2d(bn_size*growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_rate)

        self.memory_efficient = memory_efficient

    def requires_grad(self, input):
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    def forward(self, x):
        identity = x

        if not self.memory_efficient and self.requires_grad(identity):
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.dropout(x)

        return x