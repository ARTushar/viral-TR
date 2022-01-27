from utils.tp_chroms import tp_chroms_to_bed_file 
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


def generate_tp_bed_file(dataset_name, version):
	data_dir = os.path.join('matrix', dataset_name)
	ckpt_file = os.path.join(GDIR, 'cv_best_saved_models', f'version_{version}.ckpt')

	model = SimpleModel.load_from_checkpoint(ckpt_file)
	model.conv1d.run_value = 3

	trainer = pl.Trainer(
	deterministic=True,
	)

	print('\n*** *** *** for generating TP Seqs *** *** ***')
	init_metrics = trainer.test(model, datamodule=SequenceDataModule(
	os.path.join(DDIR, data_dir),
	'seq.fa',
	'out.dat',
	batch_size=512,
	for_test='both'
	), verbose=False)[0]
	print(json.dumps(init_metrics, indent=4))

	tp_chroms_to_bed_file(model.tp_chroms, data_dir, version)

if __name__ == '__main__':
    dataset_name = 'HBZ-ST1'
    version = 18
    generate_tp_bed_file(dataset_name, version)


