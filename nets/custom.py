import torch
from torch import nn
import torch.nn.functional as F
device = "cuda"

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 12, 3)
        self.conv3 = nn.Conv2d(12, 16, 5)
        self.conv4 = nn.Conv2d(16, 32, 5)
        self.conv5 = nn.Conv2d(32, 64, 7)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16*5*5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 6)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv5(x)), (2, 2))
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        self.fc1=nn.Linear(x.shape[1], 120).to(device)
        x = F.relu(self.fc1(x))
        # m=nn.Dropout(p=0.5, inplace=True)
        # x=m(x)
        x = F.relu(self.fc2(x))
        # m=nn.Dropout(p=0.5, inplace=True)
        # x=m(x)
        x = self.fc3(x)
        return x




class WeightClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-1, 1)
            module.weight.data = w






class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
        # 3 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 12, 3)
        self.conv3 = nn.Conv2d(12, 16, 5)
        self.conv4 = nn.Conv2d(16, 32, 5)
        self.conv5 = nn.Conv2d(32, 64, 7)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16*5*5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 6)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv5(x)), (2, 2))
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        self.fc1=nn.Linear(x.shape[1], 120)
        x = F.relu(self.fc1(x))
        # m=nn.Dropout(p=0.5, inplace=True)
        # x=m(x)
        x = F.relu(self.fc2(x))
        # m=nn.Dropout(p=0.5, inplace=True)
        # x=m(x)
        x = self.fc3(x)
        return x