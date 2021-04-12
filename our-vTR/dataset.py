import os
import pytorch_lightning as pl
from torch.utils.data import Dataset
from typing import Callable, Optional
from torch.utils.data.dataloader import DataLoader
from utils.splitter import splitter
from utils.transforms import transform_input, transform_label


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


class SequenceDataModule(pl.LightningDataModule):

    def __init__(self, directory, file_in, file_out, seed=4, batch_size=32):
        super().__init__()
        self.directory = directory
        self.file_in = file_in
        self.file_out = file_out
        self.batch_size = batch_size
        self.seed = seed

    def prepare_data(self):
        splitter(self.directory, self.file_in, self.file_out, self.seed)

    def setup(self, stage: Optional[str] = None):
        train_file_in = os.path.join(self.directory, 'train', self.file_in)
        train_file_out = os.path.join(self.directory, 'train', self.file_out)
        cv_file_in = os.path.join(self.directory, 'cv', self.file_in)
        cv_file_out = os.path.join(self.directory, 'cv', self.file_out)
        test_file_in = os.path.join(self.directory, 'test', self.file_in)
        test_file_out = os.path.join(self.directory, 'test', self.file_out)

        if stage == 'fit':
            self.train_data = CustomSequenceDataset(train_file_in, train_file_out, transform_input, transform_label)
            self.val_data = CustomSequenceDataset(cv_file_in, cv_file_out, transform_input, transform_label)
        
        if stage == 'test':
            self.test_data = CustomSequenceDataset(test_file_in, test_file_out, transform_input, transform_label)
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_data , batch_size=self.batch_size, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)
