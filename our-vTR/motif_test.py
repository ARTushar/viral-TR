from dataset import SequenceDataModule
from model import SimpleModel
import pytorch_lightning as pl

version = 4
model = SimpleModel.load_from_checkpoint(f'../globals/saved_models/version_{version}.ckpt')

trainer = pl.Trainer()

print('\n*** *** *** for train *** *** ***')
datamodule=SequenceDataModule(
    # '../dirs/dataset2',
    # 'SRR5241432_seq.fa',
    # 'SRR5241432_out.dat',
    '../globals/datasets/normal/normal/SRR5241432',
    'seq.fa',
    'out.dat',
    batch_size=512,
    for_test='both'
)
train_metrics = trainer.test(model, datamodule=datamodule, verbose=True)[0]