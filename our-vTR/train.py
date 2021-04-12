import time
from dataset import SequenceDataModule
from model import SimpleModel
import pytorch_lightning as pl


def main():
    data_module = SequenceDataModule('dataset', 'sequences.fa', 'wt_readout.dat', batch_size=64)
    # data_module = SequenceDataModule('dummyset', 'in.txt', 'out.txt', batch_size=64)
    model = SimpleModel()
    trainer = pl.Trainer(gpus=1, max_epochs=10)
    start_time = time.time()
    trainer.fit(model, datamodule=data_module)
    print(f'---- {time.time() - start_time} seconds ----\n\n\n')

    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    main()