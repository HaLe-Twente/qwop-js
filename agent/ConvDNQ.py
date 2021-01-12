import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd


class ConvDNQ(nn.Module):
    def __init__(self):
        super(ConvDNQ, self).__init__()

        self.conf1 = nn.Conv2d(4, 32, 8, 4)
        #self.pool1 = nn.MaxPool2d(2, 2)
        self.conf2 = nn.Conv2d(32, 64, 4, 2)
        #self.pool2 = nn.MaxPool2d(2, 2)
        self.conf3 = nn.Conv2d(64, 64, 3, 1)
        #self.pool3 = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(26624, 512)
        self.linear2 = nn.Linear(512, 8)

    def forward(self, x):
        #print(x.shape)

        x = self.conf1(x)
        x = F.relu(x)
        #print(x.shape)

        x = self.conf2(x)
        x = F.relu(x)
        #print(x.shape)

        x = self.conf3(x)
        x = F.relu(x)
        #print(x.shape)

        #x = x.view(-1, 1024)
        x = x.view(x.size()[0], -1)
        #print(x.shape)

        x = self.linear1(x)
        x = F.relu(x)
        #print(x.shape)

        x = self.linear2(x)
        return x






