import os
import json

from dataset import SequenceDataModule
from model import SimpleModel
import pytorch_lightning as pl

from utils.metrics import change_keys

GDIR = os.path.join('..', 'globals')
DDIR = os.path.join(GDIR, 'datasets')

SEED = 4
pl.seed_everything(SEED, workers=True)

data_dir = 'matrix/EBNA2-Mutu3'
version = 54
ckpt_dir = os.path.join(GDIR, 'saved_models', data_dir, f'version_{version}.ckpt')

model = SimpleModel.load_from_checkpoint(ckpt_dir)
model.eval()

trainer = pl.Trainer(
    deterministic=True,
)

print('\n*** *** *** for train *** *** ***')
train_metrics = trainer.predict(model, datamodule=SequenceDataModule(
    os.path.join(DDIR, data_dir),
    'seq.fa',
    'out.dat',
    batch_size=512,
    for_test='both'
))[0]

change_keys(train_metrics, 'train', 'test')
print(json.dumps(train_metrics, indent=4))

for i in range(5):
    print('\n*** *** *** for val *** *** ***')
    val_metrics = trainer.predict(model, datamodule=SequenceDataModule(
        os.path.join(DDIR, data_dir),
        'seq.fa',
        'out.dat',
        batch_size=512,
        for_test='val'
    ))[0]
    change_keys(val_metrics, 'val', 'test')
    print(json.dumps(val_metrics, indent=4))
