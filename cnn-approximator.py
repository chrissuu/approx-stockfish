import chess
import chess.svg
import numpy as np
import pandas as pd

from IPython.display import display,HTML
from google.colab import files
from sklearn.model_selection import train_test_split

files.upload()
!pip install -q kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d ronakbadhe/chess-evaluations
!unzip chess-evaluations.zip

!ls

#read our csv file
df = pd.read_csv('chessData.csv')

#displays head
df.head()

#randomly sample a chess position and display it
randomChessPos = df['FEN'].sample().values[0]
board = chess.Board(randomChessPos)
display(HTML(chess.svg.board(board, size=600)))

"""<============== UTILS ==============>"""

!pip install torch

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

