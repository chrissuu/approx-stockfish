import constants
from utils import FENtoVEC

import torch
import numpy as np
from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision.io import read_image

class ChessDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        #(FEN, label)
        fen = self.df.iloc[idx]['FEN']
        label = self.df.iloc[idx]['Evaluation']

        board_vec = FENtoVEC(fen)  # len 64
        board_arr = np.array(board_vec).reshape(8, 8)

        input_tensor = torch.tensor(board_arr, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            input_tensor = self.transform(input_tensor)

        return input_tensor, label_tensor
