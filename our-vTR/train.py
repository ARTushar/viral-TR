from dataset import SequenceDataModule
from model import SimpleModel
import pytorch_lightning as pl


def main():
    data_module = SequenceDataModule('dataset', 'sequences.fa', 'wt_readout.dat')
    model = SimpleModel()
    trainer = pl.Trainer()
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()