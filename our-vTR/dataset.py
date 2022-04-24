import os
import random
from typing import Callable, Optional, Tuple, List

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils import data
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import ConcatDataset

from utils.splitter import read_samples, splitter
from utils.transforms import transform_all_labels, transform_all_sequences

device = ''
with open('device.txt', 'r') as f:
    device = f.readline().strip()

WORKERS = 4 
if 'colab' in device:
    WORKERS = 2
elif 'server' in device:
    WORKERS = 6


# splitter('./dataset', 'sequences.fa', 'wt_readout.dat', 4)

class CustomSequenceDataset(Dataset):
    def __init__(self, sequences: list, labels: list, transform: Callable=None, target_transform: Callable=None, chroms=None) -> None:
        super().__init__()

        mx_len = max(len(seq) for seq in sequences)

        self.forward_sequences, self.reverse_sequences = transform(sequences, mx_len)
        self.labels = target_transform(labels)
        self.chroms = chroms

        # self.forward_sequences = self.forward_sequences[0:128]
        # self.reverse_sequences = self.reverse_sequences[0:128]
        # self.labels = self.labels[0:128]


    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, index) -> Tuple[Tensor, ...]:

        fw = self.forward_sequences[index]
        rv = self.reverse_sequences[index]
        y = self.labels[index]
        c = self.chroms[index]

        return fw, rv, y, c


class SequenceDataModule(pl.LightningDataModule):

    def __init__(self, directory, file_in, file_out, batch_size=32, for_test: str = 'test'):
        super().__init__()
        self.directory = directory
        self.file_in = file_in
        self.file_out = file_out
        self.batch_size = batch_size
        self.for_test = for_test
        # self.seed = seed # redundant

    def prepare_data(self):
        # splitter(self.directory, self.file_in, self.file_out)
        pass

    def get_all(self) -> Tuple:
        raw_file_in = os.path.join(self.directory, 'raw', self.file_in)
        raw_file_out = os.path.join(self.directory, 'raw', self.file_out)
        raw_chroms, raw_sequences, raw_labels = read_samples(raw_file_in, raw_file_out, True)
        data = list(zip(raw_sequences, raw_labels, raw_chroms))
        random.shuffle(data)
        raw_sequences = [a for a, b, c in data]
        raw_labels = [b for a, b, c in data]
        raw_chroms = [c for a, b, c in data]
        return raw_sequences, raw_labels, raw_chroms

    def setup(self, stage: Optional[str] = None):
        train_file_in = os.path.join(self.directory, 'train', self.file_in)
        train_file_out = os.path.join(self.directory, 'train', self.file_out)
        cv_file_in = os.path.join(self.directory, 'val', self.file_in)
        cv_file_out = os.path.join(self.directory, 'val', self.file_out)
        test_file_in = os.path.join(self.directory, 'test', self.file_in)
        test_file_out = os.path.join(self.directory, 'test', self.file_out)

        t_chroms, train_sequences, train_labels = read_samples(
            train_file_in, train_file_out, True)
        v_chroms, cv_sequences, cv_labels = read_samples(cv_file_in, cv_file_out, True)


        if stage == 'fit':
            if self.for_test == 'both':
                self.train_data = CustomSequenceDataset(
                 [*train_sequences, *cv_sequences], [*train_labels, *cv_labels], transform_all_sequences, transform_all_labels, [*t_chroms, *v_chroms])
            if self.for_test == 'all':
                raw_sequences, raw_labels, raw_chroms = self.get_all()
                self.train_data = CustomSequenceDataset(
                    raw_sequences, raw_labels, transform_all_sequences, transform_all_labels, raw_chroms)
            else:
                self.train_data = CustomSequenceDataset(
                    train_sequences, train_labels, transform_all_sequences, transform_all_labels, t_chroms)
            self.val_data = CustomSequenceDataset(
                cv_sequences, cv_labels, transform_all_sequences, transform_all_labels, v_chroms)

        if stage == 'test':
            if self.for_test == 'train':
                self.test_data = CustomSequenceDataset(
                 train_sequences, train_labels, transform_all_sequences, transform_all_labels, t_chroms)
            elif self.for_test == 'val':
                self.test_data = CustomSequenceDataset(
                 cv_sequences, cv_labels, transform_all_sequences, transform_all_labels, v_chroms)
            elif self.for_test == 'both':
                self.test_data = CustomSequenceDataset(
                 [*cv_sequences, *train_sequences], [*cv_labels, *train_labels], transform_all_sequences, transform_all_labels, [*v_chroms, *t_chroms])
            elif self.for_test == 'all':
                raw_sequences, raw_labels, raw_chroms = self.get_all()
                self.test_data = CustomSequenceDataset(
                    raw_sequences, raw_labels, transform_all_sequences, transform_all_labels, raw_chroms)


    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=WORKERS)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=WORKERS)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=WORKERS)

    # def predict_dataloader(self):
    #     return DataLoader(self.predict_data, batch_size=self.batch_size, num_workers=WORKERS)

    def sequence_length(self):
        return self.train_data[0][0].shape[1]


def main():
    # data_module = SequenceDataModule(
    #     'dataset1', 'SRR3101734_seq.fa', 'SRR3101734_out.dat', batch_size=64)
    data_module = SequenceDataModule('../globals/datasets/matrix/EBNA2-Mutu3', 'seq.fa', 'out.dat', batch_size=64, for_test='both')
    data_module.prepare_data()
    data_module.setup(stage='fit')
    torch.set_printoptions(threshold=10)
    for a, b, c, d in data_module.train_dataloader():
        d = list(d)
        print(a.shape)
        print(b.shape)
        print(c.shape)
        print(len(d))
        break
    print(data_module.sequence_length())


if __name__ == "__main__":
    main()
