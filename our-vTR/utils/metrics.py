import os, csv
from numpy.core.arrayprint import DatetimeFormat
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple
from utils.metric_namer import change_keys
from datetime import date

def find_metrics(
    model: LightningModule,
    trainer: Trainer,
    train_loader: DataLoader,
    val_loader: DataLoader
) -> Tuple[Dict, Dict]:

    print('\n\t\t*** *** *** calculating metrics *** *** ***')
    print('\n*** *** for train *** ***')
    train_metric = trainer.test(model, test_dataloaders=train_loader)[0]
    change_keys(train_metric, 'train', 'test')
    print('\n*** *** for validation *** ***')
    val_metric = trainer.test(model, test_dataloaders=val_loader)[0]
    change_keys(val_metric, 'val', 'test')

    return train_metric, val_metric


def log_metrics(params: Dict, metrics: Dict) -> None:
    log_dir = 'cv_params_log'

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    log_file = os.path.join(log_dir, 'results-' +
                            date.today().strftime('%d-%m-%Y') + '.csv')
    logs = {**params, **metrics}
    file_exists = os.path.isfile(log_file)
    f = open(log_file, 'a')
    dictWriter = csv.DictWriter(f, fieldnames=list(logs.keys()))
    if not file_exists:
        dictWriter.writeheader()
    dictWriter.writerow(logs)