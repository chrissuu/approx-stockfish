import chess
import chess.svg
import numpy as np
import pandas as pd
import math
import torch
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataset import random_split
from torch.utils.data.dataset import Dataset, DataLoader

from IPython.display import display, HTML
from google.colab import files
from sklearn.model_selection import train_test_split

import random

from utils import ChessDataset

ROOT = "../chess-dataset"
#read our csv file
df = pd.read_csv(f"{ROOT}/chessData.csv")

dataset = ChessDataset(df)
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # 20% for validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


