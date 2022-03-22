'''MobileNet in PyTorch.
See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.cbam1 import CBAM
device='cuda'

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    cfg = [(32, 2), 32, (64, 2), 64, (128, 2), 128, (256, 2), 256]

    def __init__(self, num_classes=6):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layers = self._make_layers(in_planes=16)
        self.linear = nn.Linear(4096, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        cbam_block1 = CBAM(gate_channels=out.shape[1]).to(device)
        out = cbam_block1(out)
        out=Block(16,32,2)(out)
        out = Block(32, 32, 1)(out)
        out = Block(32, 64, 2)(out)
        out = Block(64, 64, 1)(out)
        out = Block(64, 128, 2)(out)
        out = Block(128, 128, 1)(out)
        cbam_block2 = CBAM(gate_channels=out.shape[1]).to(device)
        out = cbam_block2(out)
        out = Block(128, 256, 2)(out)
        out = Block(256, 256, 1)(out)
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = MobileNet().to(device)
    x = torch.randn(32, 1, 300,400).to(device)
    y = net(x)
    print(y.size())

    ##