from copy import deepcopy
from pathlib import Path
from typing import Union, Optional, Callable

from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import ConcatDataset, Subset, DataLoader

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger, LoggerCollection
from dataset import CustomSequenceDataset
from utils.transforms import transform_all_sequences, transform_all_labels

from model import SimpleModel


class DataCV:
    """dataset for cross-validation."""

    def __init__(self,
                 data_dir: Union[str, Path],
                 sequence_file: str,
                 label_file: str,
                 num_workers: int = 16,
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

            yield train_loader, val_loader

    def get_dataset(self):
        """Creates and returns the complete dataset."""
        train_sequence_path = Path(self.data_dir).joinpath(
            'train',  self.sequence_file)
        train_label_path = Path(self.data_dir).joinpath(
            'train', self.label_file)
        cv_sequence_path = Path(self.data_dir).joinpath(
            'cv',  self.sequence_file)
        cv_label_path = Path(self.data_dir).joinpath('cv', self.label_file)

        train_dataset = CustomSequenceDataset(
            train_sequence_path, train_label_path, transform_all_sequences, transform_all_labels)
        cv_dataset = CustomSequenceDataset(
            cv_sequence_path, cv_label_path, transform_all_sequences, transform_all_labels)

        return ConcatDataset([train_dataset, cv_dataset])

    def get_data_labels(self):
        dataset = self.get_dataset()
        return [int(sample[2]) for sample in dataset]


class CV:
    """Cross-validation with a LightningModule."""

    def __init__(self,
                 *trainer_args,
                 **trainer_kwargs):
        super().__init__()
        self.trainer_args = trainer_args
        self.trainer_kwargs = trainer_kwargs

    @staticmethod
    def _update_logger(logger, fold_idx: int):
        if hasattr(logger, 'experiment_name'):
            logger_key = 'experiment_name'
            print('exp name ', getattr(logger, 'experiment_name'))
        elif hasattr(logger, 'name'):
            logger_key = 'name'
            print(getattr(logger, 'name'))
        else:
            raise AttributeError('The logger associated with the trainer '
                                 'should have an `experiment_name` or `name` '
                                 'attribute.')
        new_experiment_name = getattr(logger, logger_key) + f'/{fold_idx}'
        setattr(logger, logger_key, new_experiment_name)

    @staticmethod
    def update_modelcheckpoint(model_ckpt_callback, fold_idx):
        _default_filename = '{epoch}-{step}'
        _suffix = f'_fold{fold_idx}'
        if model_ckpt_callback.filename is None:
            new_filename = _default_filename + _suffix
        else:
            new_filename = model_ckpt_callback.filename + _suffix
        setattr(model_ckpt_callback, 'filename', new_filename)

    def update_logger(self, trainer: Trainer, fold_idx: int):
        if not isinstance(trainer.logger, LoggerCollection):
            _loggers = [trainer.logger]
        else:
            _loggers = trainer.logger

        # Update loggers:
        for _logger in _loggers:
            self._update_logger(_logger, fold_idx)

    def fit(self, model: LightningModule, data: DataCV):
        splits = data.get_splits()
        for fold_idx, loaders in enumerate(splits):

            print("Fold inx: ", fold_idx)

            # Clone model & instantiate a new trainer:
            _model = deepcopy(model)
            trainer = Trainer(*self.trainer_args, **self.trainer_kwargs)

            # Update loggers and callbacks:
            # self.update_logger(trainer, fold_idx)
            for callback in trainer.callbacks:
                if isinstance(callback, ModelCheckpoint):
                    self.update_modelcheckpoint(callback, fold_idx)

            # Fit:
            trainer.fit(_model, *loaders)


if __name__ == '__main__':

    # Trainer
    # NEPTUNE_PROJECT_NAME = 'viral-tf'
    # NEPTUNE_EXPERIMENT_NAME = 'cv-test'
    # neptune_logger = NeptuneLogger(project_name=NEPTUNE_PROJECT_NAME,
    #                                 offline_mode=True,
    #                                experiment_name=NEPTUNE_EXPERIMENT_NAME)

    # model_checkpoint = ModelCheckpoint(dirpath=MODEL_CHECKPOINT_DIR_PATH,
    #                                    monitor='val_acc',
    #                                    save_top_k=1,
    #                                    mode='max',
    #                                    filename='custom_model_{epoch}',)

    trainer_kwargs_ = {
                        # 'weights_summary': None,
                    #    'progress_bar_refresh_rate': 1,
                    #    'num_sanity_val_steps': 0,
                       'gpus': [0],
                       'max_epochs': 10,
                    #    'logger': neptune_logger,
                       #    'callbacks': [model_checkpoint]
                       }

    cv = CV(**trainer_kwargs_)

    # LightningModule

    params = {
        'kernel_size': 12,
        'alpha': 1e3,
        'beta': 1e-3,
        'l1_lambda': 1e-3,
        'l2_lambda': 0,
        'batch_size': 64,
        'epochs': 20
    }
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
    DATA_DIR = 'dataset'
    SEQUENCE_FILE = 'sequences.fa'
    LABEL_FILE = 'wt_readout.dat'

    # Run a 5-fold cross-validation experiment:
    sequence_data = DataCV(data_dir=DATA_DIR, sequence_file=SEQUENCE_FILE, label_file=LABEL_FILE, n_splits=5, stratify=False)

    cv.fit(model, sequence_data)
