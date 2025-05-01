'''LIBRARIES'''
import chess
import chess.svg
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from IPython.display import display, HTML
from sklearn.model_selection import train_test_split
from model import ChessCNN
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from tqdm import trange
from tqdm import tqdm

'''PYTHON DEFAULT LIBRARIES'''
import math
import random

'''CUSTOM MODULES + UTILS'''
import constants
import model

from utils import ChessDataset
from utils import get_train_test_val_dataloaders

# read our csv file and preprocess
df = pd.read_csv(f"{constants.ROOT}/chessData.csv")
df['Evaluation'] = df['Evaluation'].astype(str).str.replace('#', '')
df['Evaluation'] = df['Evaluation'].astype(float)
df = df.sample(frac=0.1)

dataset = ChessDataset(df)

train_size = int(0.5 * len(dataset))
test_size = int(0.3 * len(dataset))
val_size = int(0.1 * len(dataset))

train_size += len(dataset) - train_size - test_size - val_size # want to use all data, so <cleanly> collect leftovers

train_loader, test_loader, val_loader = get_train_test_val_dataloaders(
    train_size, 
    test_size,
    val_size,
    constants.BATCH_SIZE,
    dataset=dataset
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = ChessCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 10

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        boards, labels = batch
        boards = boards.to(device)
        labels = labels.to(device)
        # boards will have shape (batch_size, 1, 8, 8)
        optimizer.zero_grad()
        outputs = model(boards)
        outputs = outputs.squeeze()
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for boards, labels in val_loader:
          boards = boards.to(device)
          labels = labels.to(device)
          outputs = model(boards).squeeze(1)
          loss = criterion(outputs, labels)
          val_loss += loss.item() * boards.size(0)                                                                                                                               

    val_loss /= len(val_loader.dataset)
    print(f"Epoch {epoch+1}/{10}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
