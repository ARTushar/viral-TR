import os, csv
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from utils.metric_namer import change_keys
from datetime import date

def find_metrics(
    model: LightningModule,
    trainer: Trainer,
    train_loader: DataLoader,
    val_loader: DataLoader
) -> Tuple[Dict, Dict]:

    print('\n*** *** *** calculating metrics *** *** ***')
    print('for train:')
    train_metric = trainer.test(model, test_dataloaders=train_loader, verbose=False)[0]
    change_keys(train_metric, 'train', 'test')
    print('for validation:')
    val_metric = trainer.test(model, test_dataloaders=val_loader, verbose=False)[0]
    change_keys(val_metric, 'val', 'test')

    return train_metric, val_metric


def log_metrics(logs: Dict) -> None:
    log_dir = 'cv_params_log'
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    log_dir = os.path.join(log_dir, logs['data_dir'])
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    del logs['data_dir']
    del logs['sequence_file']
    del logs['label_file']

    log_file = os.path.join(
        log_dir,
        'results-' + date.today().strftime('%d-%m-%Y') + '.csv'
    )
    file_exists = os.path.isfile(log_file)
    f = open(log_file, 'a')

    headers = list(logs.keys())
    headers.insert(0, 'version')

    dictWriter = csv.DictWriter(f, fieldnames=headers)
    if file_exists:
        logs['version'] = sum(1 for _ in csv.reader(open(log_file)))
    else:
        dictWriter.writeheader()
        logs['version'] = 1


    dictWriter.writerow(logs)
