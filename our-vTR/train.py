import json
import os
import csv
import time
import random
from datetime import date
from argparse import ArgumentParser, Namespace
from typing import Dict

import pytorch_lightning as pl
from pytorch_lightning.callbacks import early_stopping
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import ConcatDataset

from dataset import CustomSequenceDataset, SequenceDataModule
from model import SimpleModel
from utils.transforms import transform_all_labels, transform_all_sequences
from utils.metrics import change_keys

SEED = 0


def train(params: Dict) -> None:
    pl.seed_everything(SEED, workers=True)
    data_module = SequenceDataModule(
        params["data_dir"],
        params["sequence_file"],
        params["label_file"],
        batch_size=params['batch_size']
    )

    early_stopper = EarlyStopping(monitor='valLoss', verbose=True)

    # trainer = pl.Trainer.from_argparse_args(
    #     args, deterministic=True, gpus=-1, auto_select_gpus=True)
    # trainer = pl.Trainer.from_argparse_args(args, deterministic=True)
    trainer = pl.Trainer(
        max_epochs=params['epochs'],
        deterministic=True,
        # gpus=-1,
        # auto_select_gpus=True,
        callbacks=[early_stopper]
    )

    model = SimpleModel(
        convolution_type=params['convolution_type'],
        kernel_size=params['kernel_size'],
        kernel_count=params['kernel_count'],
        alpha=params['alpha'],
        beta=params['beta'],
        distribution=tuple(params['distribution']),
        linear_layer_shapes=list(params['linear_layer_shapes']),
        l1_lambda=params['l1_lambda'],
        l2_lambda=params['l2_lambda'],
        lr=params['learning_rate'],
    )

    start_time = time.time()
    trainer.fit(model, datamodule=data_module)
    print(f'\n---- {time.time() - start_time} seconds ----\n\n\n')

    print('hello there:')
    print(early_stopper.best_score)
    print(early_stopper.stopped_epoch)

    print('\n*** *** *** for train *** *** ***')
    train_metrics = trainer.test(model, datamodule=SequenceDataModule(
        params["data_dir"],
        params["sequence_file"],
        params["label_file"],
        batch_size=512,
        for_test='train'
    ))[0]
    print('\n*** *** *** for val *** *** ***')
    val_metrics = trainer.test(model, datamodule=SequenceDataModule(
        params["data_dir"],
        params["sequence_file"],
        params["label_file"],
        batch_size=512,
        for_test='val'
    ))[0]
    print('\n*** *** *** for train+val *** *** ***')
    both_metrics = trainer.test(model, datamodule=SequenceDataModule(
        params["data_dir"],
        params["sequence_file"],
        params["label_file"],
        batch_size=512,
        for_test='both'
    ))[0]

    change_keys(train_metrics, 'train', 'test')
    change_keys(val_metrics, 'val', 'test')
    change_keys(both_metrics, 'both', 'test')

    version = trainer.logger.version
    extra = {
        'device': 'redwan-pc',
        'version': version,
        'seed': SEED
    }

    log_dir = os.path.join('params_log', params['data_dir'])

    del params['data_dir']
    del params['sequence_file']
    del params['label_file']

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    log_file = os.path.join(log_dir, 'results-' + date.today().strftime('%d-%m-%Y') + '.csv')
    logs = {**extra, **params, **train_metrics, **val_metrics, **both_metrics}
    file_exists = os.path.isfile(log_file)
    f = open(log_file, 'a')
    dictWriter = csv.DictWriter(f, fieldnames=list(logs.keys()))
    if not file_exists:
        dictWriter.writeheader()
    dictWriter.writerow(logs)


if __name__ == "__main__":
    params = json.load(open('parameters.json'))
    # parser = ArgumentParser()
    # parser = pl.Trainer.add_argparse_args(parser)
    # args = parser.parse_args()
    # args.__setattr__('max_epochs', params['epochs'])
    train(params)
