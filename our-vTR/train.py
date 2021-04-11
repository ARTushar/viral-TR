import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import CustomSequenceDataset
import os
from utils.transforms import transform_input, transform_label 
from fit import fit
from model import SimpleModel

class Trainer:
    def __init__(self, file_in, file_out) -> None:
        train_input = os.path.join('dataset', 'train', file_in)
        train_output = os.path.join('dataset', 'train', file_out)
        cv_input = os.path.join('dataset', 'cv', file_in)
        cv_output = os.path.join('dataset', 'cv', file_out)
        self.train_dataset = CustomSequenceDataset(train_input, train_output, transform_input, transform_label)
        self.cv_dataset = CustomSequenceDataset(cv_input, cv_output, transform_input, transform_label)
        self.model = SimpleModel()
      

    def train(self, batch_size:int=8) -> None:
        train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size)
        cv_dataloader = DataLoader(self.cv_dataset, batch_size=batch_size)
        fit(
            epochs = 10,
            model = self.model,
            loss_func = nn.BCELoss(),
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2, weight_decay=1),
            train_dl = train_dataloader,
            valid_dl = cv_dataloader
        )


def main():
    trainer = Trainer('sequences.fa', 'wt_readout.dat')
    trainer.train()

if __name__ == "__main__":
    main()