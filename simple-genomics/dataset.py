import os
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from typing import Callable, Optional
from torch.utils.data.dataloader import DataLoader
from utils.splitter import splitter, read_samples
from utils.transforms import transform_all_sequences, transform_all_labels 
from sklearn.model_selection import train_test_split



class CustomSequenceDataset(Dataset):
    def __init__(self, sequences=list, labels=list,  transform: Callable=None, target_transform: Callable=None) -> None:
        super().__init__()
        self.forward_sequences, self.reverse_sequences = transform(sequences)
        self.labels = target_transform(labels)

    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, index) -> dict:

        fw = self.forward_sequences[index]
        rv = self.reverse_sequences[index]
        y = self.labels[index]

        return fw, rv, y


class SequenceDataModule(pl.LightningDataModule):

    def __init__(self, directory, file_in, file_out, seed=4, batch_size=32):
        super().__init__()
        self.directory = directory
        self.file_in = file_in
        self.file_out = file_out
        self.batch_size = batch_size
        self.seed = seed

    def prepare_data(self):
        sequences_file = os.path.join(self.directory, 'raw', self.file_in)
        labels_file = os.path.join(self.directory, 'raw', self.file_out)
        self.sequences, self.labels = read_samples(sequences_file, labels_file)

    def setup(self, stage: Optional[str] = None):

        X_train, X_test, y_train, y_test = train_test_split(self.sequences, self.labels, test_size=0.1, random_state=self.seed)

        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=self.seed)

        if stage == 'fit':
            self.train_data = CustomSequenceDataset(X_train, y_train, transform_all_sequences, transform_all_labels)
            self.val_data = CustomSequenceDataset(X_valid, y_valid, transform_all_sequences, transform_all_labels)
        
        if stage == 'test':
            self.test_data = CustomSequenceDataset(X_test, y_test, transform_all_sequences, transform_all_labels)
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True , num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_data , batch_size=self.batch_size, num_workers=2)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)


def main():
    data_module = SequenceDataModule('dataset', 'sequences.txt', 'labels.txt', batch_size=64)
    data_module.prepare_data()
    data_module.setup(stage="fit")
    torch.set_printoptions(threshold=10)
    for a, b, c in data_module.train_dataloader():
        print(a)
        print(b)
        print(c)
        break


if __name__ == "__main__":
    main()