from dataset import SequenceDataModule
import os
from model import SimpleModel
from utils.motif import make_motif
import pytorch_lightning as pl

version = 13
model = SimpleModel.load_from_checkpoint(f'../dirs/saved_models/version_{version}.ckpt')


trainer = pl.Trainer()

print('\n*** *** *** for train *** *** ***')
train_metrics = trainer.test(model, datamodule=SequenceDataModule(
    '../dirs/dataset1',
    'SRR3101734_seq.fa',
    'SRR3101734_out.dat',
    batch_size=512,
    for_test='train'
), verbose=True)[0]