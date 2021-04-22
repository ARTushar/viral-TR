import json
import time
from argparse import ArgumentParser, Namespace
from typing import Dict

import pytorch_lightning as pl

from dataset import SequenceDataModule
from model import SimpleModel

SEED = 70
pl.seed_everything(SEED)


def main(args: Namespace, params: Dict) -> None:
    data_module = SequenceDataModule(
        'dataset',
        'sequences.fa',
        'wt_readout.dat',
        batch_size=params['batch_size']
    )

    # trainer = pl.Trainer.from_argparse_args(args, deterministic=True, gpus=-1, auto_select_gpus=True)
    trainer = pl.Trainer.from_argparse_args(args, deterministic=True)

    model = SimpleModel(
        convolution_type=params['convolution_type'],
        kernel_size=params['kernel_size'],
        kernel_count=params['kernel_count'],
        alpha=params['alpha'],
        beta=params['beta'],
        distribution=params['distribution'],
        linear_layer_shapes=params['linear_layer_shapes'],
        l1_lambda=params['l1_lambda'],
        l2_lambda=params['l2_lambda'],
        lr=params['learning_rate'],
    )

    start_time = time.time()
    trainer.fit(model, datamodule=data_module)
    print(f'\n---- {time.time() - start_time} seconds ----\n\n\n')

    # trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    params = json.load(open('parameters.json'))

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args.__setattr__('max_epochs', params['epochs'])

    main(args, params)

