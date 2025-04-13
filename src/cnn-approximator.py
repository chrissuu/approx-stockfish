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

from IPython.display import display, HTML
from google.colab import files
from sklearn.model_selection import train_test_split

import random

ROOT = "../chess-dataset"
#read our csv file
df = pd.read_csv(f"{ROOT}/chessData.csv")





