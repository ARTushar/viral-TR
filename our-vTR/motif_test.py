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

kernel_count = 16
output_file = 'HBZ-ST1-ranks.txt'
data_dir = 'matrix/HBZ-ST1'
version = 18
ckpt_file = os.path.join(GDIR, 'cv_best_saved_models', f'version_{version}.ckpt')

model = SimpleModel.load_from_checkpoint(ckpt_file)
model.conv1d.run_value = 3

trainer = pl.Trainer(
    deterministic=True,
)

print('\n*** *** *** for train *** *** ***')
init_metrics = trainer.test(model, datamodule=SequenceDataModule(
    os.path.join(DDIR, data_dir),
    'seq.fa',
    'out.dat',
    batch_size=512,
    for_test='both'
), verbose=False)[0]
print(json.dumps(init_metrics, indent=4))

init = init_metrics['testAccuracy']
base = 0.5

off = [12, 4]
for i in range(len(off)):
    off[i] -= 1
model.nullify(off)

print('\n*** *** *** for train *** *** ***')
test_metrics = trainer.test(model, datamodule=SequenceDataModule(
    os.path.join(DDIR, data_dir),
    'seq.fa',
    'out.dat',
    batch_size=512,
    for_test='both'
), verbose=False)[0]
print(json.dumps(test_metrics, indent=4))

cur = test_metrics['testAccuracy']

print('\n\n------')
print(f'=== {100 * (cur - init):.2f} {100 * (cur - init) / (init - base):.2f}')
print('------\n\n')

assert False

motif_rank = []

for i in range(kernel_count):
    print('\n--------------')
    print(f'nullifying {i}')
    model.nullify([i])
    test_metrics = trainer.test(model, datamodule=SequenceDataModule(
        os.path.join(DDIR, data_dir),
        'seq.fa',
        'out.dat',
        batch_size=512,
        for_test='both'
    ), verbose=False)[0]
    for key in test_metrics.keys():
        test_metrics[key] -= init_metrics[key]
    print(json.dumps(test_metrics, indent=4))
    print('--------------\n')
    cur = test_metrics['testAccuracy']
    motif_rank.append((
        100 * cur,
        100 * cur / (init - base),
        i + 1
    ))

motif_rank.sort()

with open(output_file, 'w') as f:
    print(f'initial acc: {init}\n', file=f)
    for acc, rel, no in motif_rank:
        print(f'{no:2}: {acc:.2f} {rel:.2f}', file=f)


