import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conf1 = nn.Conv2d(4, 32, 8, 4)
        self.conf2 = nn.Conv2d(32, 64, 4, 2)
        self.conf3 = nn.Conv2d(64, 64, 3, 1)
        self.linear1 = nn.Linear(26624, 512)
        self.linear2 = nn.Linear(512, 9)

    def forward(self, x):

        x = self.conf1(x)
        x = F.relu(x)

        x = self.conf2(x)
        x = F.relu(x)

        x = self.conf3(x)
        x = F.relu(x)

        x = x.view(x.size()[0], -1)

        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)
        return x
