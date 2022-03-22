'''MobileNet in PyTorch.
See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.pool0 = torch.nn.MaxPool2d(3, stride=stride, padding=1)
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        y = self.pool0(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        pool = torch.nn.MaxPool2d(3, stride=1, padding=1)
        out = pool(out)

        out = F.relu(torch.cat((out, y), dim=1))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [(32, 2), 32, (64, 2), 64, (128, 2), 128, (256, 2), 256]

    def __init__(self, num_classes=2):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layers = self._make_layers(in_planes=16)
        self.linear = nn.Linear(46848, 2)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes + in_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        pool = torch.nn.MaxPool2d(3, stride=1, padding=1)
        out = torch.cat((pool(x), x), dim=1)
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.layers(out)
        pool = torch.nn.MaxPool2d(3)
        out = pool(out)
        out = out.view(out.size(0), -1)
        # drop = nn.Dropout2d(p=0.8)
        # out=drop(out)
        # print(out.shape)
        out = self.linear(out)
        return out


def test():
    net = MobileNet()
    x = torch.randn(32, 1, 300, 400)
    y = net(x)
    print(y.size())
test()