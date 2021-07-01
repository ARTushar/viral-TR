from dataset import SequenceDataModule
import os
from model import SimpleModel
from utils.motif import make_motif
import pytorch_lightning as pl

version = 13
model = SimpleModel.load_from_checkpoint(f'../dirs/saved_models/version_{version}.ckpt')


trainer = pl.Trainer()

print('\n*** *** *** for train *** *** ***')
datamodule=SequenceDataModule(
    # '../dirs/dataset2',
    # 'SRR5241432_seq.fa',
    # 'SRR5241432_out.dat',
    'SRR5241430',
    'seq.fa',
    'out.dat',
    batch_size=512,
    for_test='both'
)
train_metrics = trainer.test(model, datamodule=datamodule, verbose=True)[0]