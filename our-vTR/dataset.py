import os
from typing import Callable, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from utils.splitter import splitter
from utils.transforms import transform_all_labels, transform_all_sequences

WORKERS = 4


# splitter('./dataset', 'sequences.fa', 'wt_readout.dat', 4)

class CustomSequenceDataset(Dataset):
    def __init__(self, file_in: str, file_out: str, transform: Callable=None, target_transform: Callable=None) -> None:
        super().__init__()
        sequences, labels = [], []
         
        with open(file_in, 'r') as ifile:
            sequences = [line.strip() for line in ifile]

        with open(file_out, 'r') as ofile:
            labels = [line.strip() for line in ofile]

        self.forward_sequences, self.reverse_sequences = transform(sequences)
        self.labels = target_transform(labels)

        self.forward_sequences = self.forward_sequences[0:128]
        self.reverse_sequences = self.reverse_sequences[0:128]
        self.labels = self.labels[0:128]


    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, index) -> Tuple[Tensor, ...]:

        fw = self.forward_sequences[index]
        rv = self.reverse_sequences[index]
        y = self.labels[index]

        return fw, rv, y


class SequenceDataModule(pl.LightningDataModule):

    def __init__(self, directory, file_in, file_out, batch_size=32):
        super().__init__()
        self.directory = directory
        self.file_in = file_in
        self.file_out = file_out
        self.batch_size = batch_size
        # self.seed = seed # redundant

    def prepare_data(self):
        splitter(self.directory, self.file_in, self.file_out)

    def setup(self, stage: Optional[str] = None):
        train_file_in = os.path.join(self.directory, 'train', self.file_in)
        train_file_out = os.path.join(self.directory, 'train', self.file_out)
        cv_file_in = os.path.join(self.directory, 'cv', self.file_in)
        cv_file_out = os.path.join(self.directory, 'cv', self.file_out)
        test_file_in = os.path.join(self.directory, 'test', self.file_in)
        test_file_out = os.path.join(self.directory, 'test', self.file_out)

        if stage == 'fit':
            self.train_data = CustomSequenceDataset(train_file_in, train_file_out, transform_all_sequences, transform_all_labels)
            self.val_data = CustomSequenceDataset(cv_file_in, cv_file_out, transform_all_sequences, transform_all_labels)
        
        if stage == 'test':
            self.test_data = CustomSequenceDataset(test_file_in, test_file_out, transform_all_sequences, transform_all_labels)
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True , num_workers=WORKERS)

    def val_dataloader(self):
        return DataLoader(self.val_data , batch_size=self.batch_size, num_workers=WORKERS)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=WORKERS)


def main():
    data_module = SequenceDataModule('dataset', 'sequences.fa', 'wt_readout.dat', batch_size=64)
    data_module.setup(stage="fit")
    torch.set_printoptions(threshold=10)
    for a, b, c in data_module.train_dataloader():
        print(a)
        print(b)
        print(c)
        break


if __name__ == "__main__":
    main()
