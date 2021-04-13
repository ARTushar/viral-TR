from argparse import ArgumentParser
import time

from dataset import SequenceDataModule
from model import SimpleModel
import pytorch_lightning as pl

SEED = 70 
pl.seed_everything(SEED)

def main(args):
    data_module = SequenceDataModule('dataset', 'sequences.fa', 'wt_readout.dat', batch_size=64)
    # data_module = SequenceDataModule('dummyset', 'in.txt', 'out.txt', batch_size=64)
    model = SimpleModel()
    trainer = pl.Trainer.from_argparse_args(args, deterministic=True, gpus=-1, auto_select_gpus=True)
    start_time = time.time()
    trainer.fit(model, datamodule=data_module)
    print(f'\n---- {time.time() - start_time} seconds ----\n\n\n')

    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)