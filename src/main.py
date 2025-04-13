'''LIBRARIES'''
import chess
import chess.svg
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from IPython.display import display, HTML
from sklearn.model_selection import train_test_split

'''PYTHON DEFAULT LIBRARIES'''
import math
import random

'''CUSTOM MODULES + UTILS'''
import constants
import model

from utils import ChessDataset
from utils import get_train_test_val_dataloaders

#read our csv file
df = pd.read_csv(f"{constants.ROOT}/chessData.csv")
print(df.head())
dataset = ChessDataset(df)

train_size = int(0.5 * len(dataset))
test_size = int(0.3 * len(dataset))
val_size = int(0.1 * len(dataset))

train_size += len(dataset) - train_size - test_size - val_size # want to use all data, so <cleanly> collect leftovers

train_dl, test_dl, val_dl = get_train_test_val_dataloaders(
    train_size, 
    test_size,
    val_size,
    constants.BATCH_SIZE,
    dataset=dataset
)

# model = model.ChessCNN()
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# for epoch in range(10):
#     # Training
#     model.train()
#     train_loss = 0.0
#     for boards, labels in train_loader:
#         # boards will have shape (batch_size, 1, 8, 8)
#         optimizer.zero_grad()
#         outputs = model(boards)
#         outputs = outputs.squeeze()
#         loss = criterion(outputs, labels)

#         loss.backward()
#         optimizer.step()

#     train_loss /= len(train_loader)

#     # Validation phase
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for boards, labels in val_loader:
#           outputs = model(boards).squeeze(1)
#           loss = criterion(outputs, labels)
#           val_loss += loss.item() * boards.size(0)

#     val_loss /= len(val_loader.dataset)
#     print(f"Epoch {epoch+1}/{10}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
