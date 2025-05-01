import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.io import read_image
from torch.utils.data.dataset import random_split

#FENtoVEC function from https://www.kaggle.com/code/gabrielhaselhurst/chess-dataset/notebook
# Outputs array of size 64 (NEED TO CONVERT TO 8x8)
def FENtoVEC (FEN):
    pieces = {"r":5,"n":3,"b":3.5,"q":9.5,"k":20,"p":1,"R":-5,"N":-3,"B":-3.5,"Q":-9.5,"K":-20,"P":-1}
    FEN = list(str(FEN.split()[0]))
    VEC = []
    for i in range(len(FEN)):
        if FEN[i] == "/":
            continue
        if FEN[i] in pieces:
            VEC.append(pieces[FEN[i]])
        else:
            em = [VEC.append(0) for i in range(int(FEN[i]))]

    # print(VEC)
    # print(type(VEC))

    return VEC

class ChessDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        #(FEN, label)
        fen = self.df.iloc[idx]['FEN']
        label = float(self.df.iloc[idx]['Evaluation'])


        board_vec = FENtoVEC(fen)  # len 64
        board_arr = np.array(board_vec).reshape(8, 8)

        input_tensor = torch.tensor(board_arr, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            input_tensor = self.transform(input_tensor)

        return input_tensor, label_tensor


def get_train_test_val_dataloaders(
        train_size, 
        test_size, 
        val_size,
        batch_size,
        dataset):
    
    train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader