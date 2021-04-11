import math
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import random


# splitter('./dataset', 'sequences.fa', 'wt_readout.dat', 4)

class CustomSequenceDataset(Dataset):
    def __init__(self, dir_from, file_in, file_out, transform) -> None:
        super().__init__()
        print(dir_from, file_in, file_out, transform)
