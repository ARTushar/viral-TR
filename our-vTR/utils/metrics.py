import os, csv
from pytorch_lightning import LightningModule, Trainer
import torch
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from datetime import date

from typing import Any, Dict

def change_keys(d: Dict[str, Any], add, sub):
    for key in list(d.keys()):
        d[add + key[len(sub):]] = d.pop(key)

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


def log_metrics(directory: str, logs: Dict) -> None:
    log_dir = os.path.join(directory, 'cv_params_log', logs['data_dir'])
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    del logs['data_dir']
    del logs['sequence_file']
    del logs['label_file']

    log_file = os.path.join(
        log_dir,
        'results-' + date.today().strftime('%d-%m-%Y') + '.csv'
    )
    file_exists = os.path.isfile(log_file)

    with open(log_file, 'a') as f:
        headers = list(logs.keys())

        dictWriter = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            dictWriter.writeheader()

        dictWriter.writerow(logs)



if __name__ == '__main__':
    d = {"ami": 2, "atumi": 1}
    change_keys(d, 'hi', 'a')
    print(d)
    # logits = torch.Tensor([[.4, .4], [.1, .9], [.4, .6]])
    # y = torch.Tensor([[1, 0], [1, 0], [0, 1]])
    # chroms = ['chrom_1', 'chrom_2', 'chrom_3']
