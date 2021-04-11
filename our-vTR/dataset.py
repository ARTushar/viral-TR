import math
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import random
from typing import Callable


# splitter('./dataset', 'sequences.fa', 'wt_readout.dat', 4)

class CustomSequenceDataset(Dataset):
    def __init__(self, file_in: str, file_out: str, transform: Callable=None, target_transform: Callable=None) -> None:
        super().__init__()
        self.sequences = []
        self.labels = []
        self.transform = transform
        self.target_transform = target_transform
         
        with open(file_in, 'r') as ifile:
            for line in ifile:
                self.sequences.append(line.strip())

        with open(file_out, 'r') as ofile:
            for line in ofile:
                self.labels.append(line.strip())


    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, index) -> dict:

        x = self.sequences[index]
        y = self.labels[index]

        if self.transform:
            fw, rv = self.transform(x)

        if self.target_transform:
            y = self.target_transform(y)

        return fw.T, rv.T, y.T
