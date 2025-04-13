import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader

class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
      x = self.conv1(x)
      x = F.relu(self.pool(x))
      x = self.conv2(x)
      x = F.relu(self.pool(x))
      x = x.view(x.size(0), -1)
      x = F.relu(self.fc1(x))
      x = self.fc2(x)
      return x
      # x = F.relu(self.fc1(x))
      # #first conv block
      #   x = F.relu(self.conv1(x))
      #   x = self.pool(x)
      # #second conv bloc
      #   x = F.relu(self.conv2(x))
      #   x = self.pool(x)
      # #third conv bloc
      #   x = F.relu(self.conv3(x))
      #   # x = self.pool(x)

      # #flatten
      #   x = x.view(x.size(0), -1)
      #   x = F.relu(self.fc1(x))
      #   x = self.fc2(x)
        # return x



