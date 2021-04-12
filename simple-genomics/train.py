from dataset import SequenceDataModule
from model import SimpleModel
import pytorch_lightning as pl


def main():
    data_module = SequenceDataModule('dataset', 'sequences.txt', 'labels.txt', batch_size=64)
    model = SimpleModel()
    trainer = pl.Trainer(gpus=1, max_epochs=50)
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()