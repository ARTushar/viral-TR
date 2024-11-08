import json
import os
import csv
import time
import random
from datetime import date
from argparse import ArgumentParser, Namespace
from typing import Dict

import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import early_stopping
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import ConcatDataset

from dataset import CustomSequenceDataset, SequenceDataModule
from model import SimpleModel
from utils.motif import make_motif
from utils.splitter import read_samples
from utils.transforms import transform_all_labels, transform_all_sequences
from utils.metrics import change_keys
from utils.predictor import calc_metrics

GDIR = os.path.join('..', 'globals')
DDIR = os.path.join(GDIR, 'datasets')

device = ''
with open('device.txt', 'r') as f:
    device = f.readline().strip()

# SEED = random.randint(0, 100)
SEED = 4

train_on = 'both'

def train(params: Dict) -> None:
    pl.seed_everything(SEED, workers=True)
    data_module = SequenceDataModule(
        os.path.join(DDIR, params["data_dir"]),
        params["sequence_file"],
        params["label_file"],
        batch_size=params['batch_size'],
        for_test=train_on
    )

    # early_stopper = EarlyStopping(monitor='valLoss')
    early_stopper = EarlyStopping(monitor='valAccuracy', mode='max', patience=100)

    # trainer = pl.Trainer.from_argparse_args(
    #     args, deterministic=True, gpus=-1, auto_select_gpus=True)
    # trainer = pl.Trainer.from_argparse_args(args, deterministic=True)

    in_colab = ('colab' in device or 'server' in device)

    logger = loggers.TensorBoardLogger(os.path.join(GDIR, 'lightning_logs'))
    trainer = pl.Trainer(
        max_epochs=params['epochs'],
        deterministic=True,
        # gpus=(-1 if in_colab else None),
        gpus=-1,
        auto_select_gpus=in_colab,
        callbacks=[early_stopper],
        logger=logger
    )

    model = SimpleModel(
        convolution_type=params['convolution_type'],
        kernel_size=params['kernel_size'],
        kernel_count=params['kernel_count'],
        alpha=params['alpha'],
        beta=params['beta'],
        distribution=tuple(params['distribution']),
        pool_type=params['pool_type'],
        linear_layer_shapes=list(params['linear_layer_shapes']),
        l1_lambda=params['l1_lambda'],
        l2_lambda=params['l2_lambda'],
        dropout_p=params['dropout_p'],
        lr=params['learning_rate'],
    )

    start_time = time.time()
    trainer.fit(model, datamodule=data_module)
    print(f'\n---- {time.time() - start_time} seconds ----\n\n\n')

    epochs = early_stopper.stopped_epoch - early_stopper.patience
    if early_stopper.stopped_epoch == 0:
        epochs = - params['epochs']
    params['epochs'] = epochs
    params['early_stopping_criteria'] = early_stopper.monitor
    params['best_score'] = float(early_stopper.best_score)

    print('\n*** *** *** for train *** *** ***')
    train_metrics = trainer.test(model, datamodule=SequenceDataModule(
        os.path.join(DDIR, params['data_dir']),
        params['sequence_file'],
        params['label_file'],
        batch_size=512,
        for_test=train_on
    ), verbose=False)[0]
    print('\n*** *** *** for val *** *** ***')
    val_metrics = trainer.test(model, datamodule=SequenceDataModule(
        os.path.join(DDIR, params['data_dir']),
        params['sequence_file'],
        params['label_file'],
        batch_size=512,
        for_test='val'
    ), verbose=False)[0]

    # print('\n*** *** *** for train+val *** *** ***')
    # both_metrics = trainer.test(model, datamodule=SequenceDataModule(
    #     os.path.join(DDIR, params['data_dir']),
    #     params['sequence_file'],
    #     params['label_file'],
    #     batch_size=512,
    #     for_test='both'
    # ), verbose=False)[0]

    # other_datasets = ['matrix/HBZ-ST1']
    # for odset in other_datasets:
    #     print('\n*** *** *** for another set: {} *** *** ***'.format(odset))
    #     trainer.test(model, datamodule=SequenceDataModule(
    #         '../globals/datasets/' + odset,
    #         params['sequence_file'],
    #         params['label_file'],
    #         batch_size=512,
    #         for_test='all'
    #     ), verbose=True)

    change_keys(train_metrics, 'train', 'test')
    change_keys(val_metrics, 'val', 'test')
    # change_keys(both_metrics, 'both', 'test')

    print(json.dumps(train_metrics, indent=4))
    print(json.dumps(val_metrics, indent=4))

    train_in = os.path.join(DDIR, params['data_dir'], 'train', params['sequence_file'])
    train_out = os.path.join(DDIR, params['data_dir'], 'train', params['label_file'])
    val_in = os.path.join(DDIR, params['data_dir'], 'val', params['sequence_file'])
    val_out = os.path.join(DDIR, params['data_dir'], 'val', params['label_file'])

    train_results = calc_metrics(model, train_in, train_out)
    val_results = calc_metrics(model, val_in, val_out)

    version = trainer.logger.version
    extra = {
        'device': device,
        'version': version,
        'seed': SEED
    }

    model_dir = os.path.join(GDIR, 'saved_models', params['data_dir'])
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    saved_file = os.path.join(model_dir, f'version_{version}.ckpt')
    trainer.save_checkpoint(saved_file)

    json_dir = os.path.join(GDIR, 'json_logs', params['data_dir'], 'version' + str(version))
    if not os.path.isdir(json_dir):
        os.makedirs(json_dir)

    with open(os.path.join(json_dir, 'train.json'), 'w') as f:
        json.dump(train_results, f, indent=4)
    with open(os.path.join(json_dir, 'validation.json'), 'w') as f:
        json.dump(val_results, f, indent=4)

    logo_dir = os.path.join(GDIR, 'logos', params['data_dir'], str(version))
    if not os.path.isdir(logo_dir):
        os.makedirs(logo_dir)
    make_motif(logo_dir, model.get_probabilities(), params['distribution'])

    log_dir = os.path.join(GDIR, 'params_log', params['data_dir'])

    del params['data_dir']
    del params['sequence_file']
    del params['label_file']
    if 'n_splits' in params:
        del params['n_splits']
    if 'stratify' in params:
        del params['stratify']

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, 'results-' + date.today().strftime('%d-%m-%Y') + '.csv')
    logs = {**extra, **params, **train_metrics, **val_metrics}
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

