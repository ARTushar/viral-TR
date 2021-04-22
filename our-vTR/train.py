from typing import Dict
import time
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl

from dataset import SequenceDataModule
from model import SimpleModel

SEED = 70
pl.seed_everything(SEED)

params = {
    'kernel_size': 12,
    'alpha': 1e3,
    'beta': 1e-3,
    'l1_lambda': 1e-3,
    'l2_lambda': 0,
    'batch_size': 64,
    'epochs': 1
}

def log_result() -> None:
    with open('param_log.txt', 'w') as f:
        f.write(str(params) + ' = 1')


def main(args: Namespace) -> None:
    data_module = SequenceDataModule(
        'dataset',
        'sequences.fa',
        'wt_readout.dat',
        batch_size=params['batch_size']
    )

    # trainer = pl.Trainer.from_argparse_args(args, deterministic=True, gpus=-1, auto_select_gpus=True)
    trainer = pl.Trainer.from_argparse_args(args, deterministic=True)

    model = SimpleModel(
        seq_length=156,
        kernel_size=params['kernel_size'],
        alpha=params['alpha'],
        beta=params['beta'],
        distribution=(0.3, 0.2, 0.2, 0.3),
        l1_lambda=params['l1_lambda'],
        l2_lambda=params['l2_lambda'],
        lr=1e-3
    )

    start_time = time.time()
    trainer.fit(model, datamodule=data_module)
    print(f'\n---- {time.time() - start_time} seconds ----\n\n\n')

    log_result()

    # trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args.__setattr__('max_epochs', params['epochs'])

    main(args)
