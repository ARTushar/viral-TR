from copy import deepcopy
import json
from logging import log
import os
import time
import random
from pathlib import Path
from typing import Dict, List, Union, Optional, Callable, Tuple

from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils import data
from torch.utils.data import ConcatDataset, Subset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger
from dataset import CustomSequenceDataset, SequenceDataModule
from utils.metric_namer import change_keys
from utils.metrics import find_metrics, log_metrics
from utils.splitter import read_samples
from utils.transforms import transform_all_sequences, transform_all_labels
from utils.tb_aggregator import write_reduced_tb_events

from model import SimpleModel


class DataCV:
    """dataset for cross-validation."""

    def __init__(self,
                 data_dir: Union[str, Path],
                 sequence_file: str,
                 label_file: str,
                 num_workers: int = 4,
                 batch_size: int = 32,
                 n_splits: int = 5,
                 stratify: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.sequence_file = sequence_file
        self.label_file = label_file
        self.num_workers = num_workers
        self.batch_size = batch_size

        # Cross-validation
        self.n_splits = n_splits
        self.stratify = stratify

    def get_splits(self):
        if self.stratify:
            labels = self.get_data_labels()
            cv_ = StratifiedKFold(n_splits=self.n_splits)
        else:
            labels = None
            cv_ = KFold(n_splits=self.n_splits)

        dataset = self.get_dataset()
        n_samples = len(dataset)
        for train_idx, val_idx in cv_.split(X=range(n_samples), y=labels):
            _train = Subset(dataset, train_idx)
            train_loader = DataLoader(dataset=_train,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=self.num_workers)

            _val = Subset(dataset, val_idx)
            val_loader = DataLoader(dataset=_val,
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    num_workers=self.num_workers)

            train_test_loader = DataLoader(dataset=_train,
                                      batch_size=512,
                                      shuffle=False,
                                      num_workers=self.num_workers)

            val_test_loader = DataLoader(dataset=_val,
                                    batch_size=512,
                                    shuffle=False,
                                    num_workers=self.num_workers)
            yield train_loader, val_loader, train_test_loader, val_test_loader


    def read_full_train_data(self):
        train_sequence_path = os.path.join(self.data_dir, 'train',  self.sequence_file)
        train_label_path = os.path.join(self.data_dir, 'train', self.label_file)
        cv_sequence_path = os.path.join(self.data_dir, 'cv',  self.sequence_file)
        cv_label_path = os.path.join(self.data_dir, 'cv', self.label_file)

        train_sequences, train_labels = read_samples(train_sequence_path, train_label_path)
        cv_sequences, cv_labels = read_samples(cv_sequence_path, cv_label_path)

        return [*train_sequences, *cv_sequences], [*train_labels, *cv_labels],


    def get_dataset(self):
        """Creates and returns the complete dataset."""
        sequences, labels = self.read_full_train_data()
        train_dataset = CustomSequenceDataset(
            sequences,
            labels,
            transform_all_sequences,
            transform_all_labels
        )
        return train_dataset

    def get_data_labels(self):
        _, labels = self.read_full_train_data()
        return labels


class CV:
    """Cross-validation with a LightningModule."""

    def __init__(self,
                log_dir: str,
                 *trainer_args,
                 **trainer_kwargs):
        super().__init__()
        self.log_dir = log_dir
        self.trainer_args = trainer_args
        self.trainer_kwargs = trainer_kwargs
        self.fitted = False # to restrict more than one self.fit() call

    # @staticmethod
    # def _update_logger(logger, fold_idx: int):
    #     if hasattr(logger, 'experiment_name'):
    #         logger_key = 'experiment_name'
    #         print('exp name ', getattr(logger, 'experiment_name'))
    #     elif hasattr(logger, 'name'):
    #         logger_key = 'name'
    #         print(getattr(logger, 'name'))
    #     else:
    #         raise AttributeError('The logger associated with the trainer '
    #                              'should have an `experiment_name` or `name` '
    #                              'attribute.')
    #     new_experiment_name = getattr(logger, logger_key) + f'_v{fold_idx}'

    #     print('got', logger_key, new_experiment_name)

    #     setattr(logger, logger_key, new_experiment_name)

    @staticmethod
    def update_modelcheckpoint(model_ckpt_callback, fold_idx):
        _default_filename = '{epoch}-{step}'
        _suffix = f'_fold{fold_idx}'
        if model_ckpt_callback.filename is None:
            new_filename = _default_filename + _suffix
        else:
            new_filename = model_ckpt_callback.filename + _suffix
        setattr(model_ckpt_callback, 'filename', new_filename)

    # def update_logger(self, trainer: Trainer, fold_idx: int):
    #     if not isinstance(trainer.logger, LoggerCollection):
    #         _loggers = [trainer.logger]
    #     else:
    #         _loggers = trainer.logger

    #     # Update loggers:
    #     for _logger in _loggers:
    #         self._update_logger(_logger, fold_idx)

    def version(self):
        for i in range(1, len(self.versions)):
            if self.versions[i] != self.versions[i-1]:
                print('CV: all versions not same: ', self.versions)
        return max(self.versions)


    def fit(self, model: LightningModule, datamodule: DataCV):
        assert not self.fitted, 'Only one call to fit() allowed per CV instance'
        self.fitted = True

        splits = datamodule.get_splits()
        avg_metrics = {}

        self.versions = [] 

        for fold_idx, (train_loader, val_loader, train_test_loader, val_test_loader) in enumerate(splits):

            print(''.join('=' for _ in range(100)), end='\n\n')

            # Clone model & instantiate a new trainer:
            _model = deepcopy(model)
            trainer = Trainer(*self.trainer_args, **self.trainer_kwargs)
            trainer.logger = TensorBoardLogger(save_dir=self.log_dir, name=f'fold_{fold_idx}')

            # Update loggers and callbacks:
            # self.update_logger(trainer, fold_idx)
            for callback in trainer.callbacks:
                if isinstance(callback, ModelCheckpoint):
                    self.update_modelcheckpoint(callback, fold_idx)

            # Fit:
            trainer.fit(_model, train_loader, val_loader)

            self.versions.append(trainer.logger.version)

            for metrics in find_metrics(_model, trainer, train_test_loader, val_test_loader):
                for key, value in metrics.items():
                    if key not in avg_metrics:
                        avg_metrics[key] = 0
                    avg_metrics[key] += value

            print(f'\n{datamodule.n_splits}-fold iteration {fold_idx}:')
            for key, value in avg_metrics.items():
                print('| {:>15} | {:.4f} |'.format(key, value/(fold_idx + 1)))

        for key in list(avg_metrics.keys()):
            avg_metrics[key] /= datamodule.n_splits
        return avg_metrics


def run_cv(params, seed: int = random.randint(1, 1000)):
    print('using seed:', seed)
    pl.seed_everything(seed)

    data_module = DataCV(
        data_dir=params['data_dir'],
        sequence_file=params['sequence_file'],
        label_file=params['label_file'],
        batch_size=params['batch_size'],
        n_splits=params['n_splits'],
        stratify=params['stratify'],
        # num_workers=2
    )

    trainer_kwargs_ = {
        # 'weights_summary': None,
        # 'progress_bar_refresh_rate': 1,
        # 'num_sanity_val_steps': 0,
        'max_epochs': params['epochs'],
        # 'deterministic': True,
        # 'gpus': -1,
        # 'auto_select_gpus': True
        # 'callbacks': [model_checkpoint]
    }

    k = params['n_splits']
    log_dir = os.path.join(f'{k}_fold_lightning_logs', params['log_dir'])
    avg_dir = os.path.join(f'{k}_fold_average_logs', params['log_dir'])

    cv = CV(log_dir=log_dir, **trainer_kwargs_)

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
    avg_metrics = cv.fit(model, datamodule=data_module)
    print(f'\n---- {time.time() - start_time} seconds ----\n\n\n')

    version = cv.version()

    log_metrics({'version': version, 'seed': seed, **params, **avg_metrics})

    write_reduced_tb_events(
        os.path.join(log_dir, 'fold_*', 'version_' + str(version)),
        os.path.join(avg_dir, 'version_' + str(version))
    )


if __name__ == '__main__':

    params = json.load(open('parameters.json'))

    # model_checkpoint = ModelCheckpoint(dirpath=MODEL_CHECKPOINT_DIR_PATH,
    #                                    monitor='val_acc',
    #                                    save_top_k=1,
    #                                    mode='max',
    #                                    filename='custom_model_{epoch}',)

    run_cv(params)
