from typing import Dict

import json

from utils.splitter import read_samples
from utils.transforms import transform_all_sequences
import pytorch_lightning as pl

def extend(d: Dict):
    d['correct'] = d['tp'] + d['tn']
    d['incorrect'] = d['fp'] + d['fn']
    d['total'] = d['correct'] + d['incorrect']
    d['accuracy'] = d['correct'] / d['total']

def calc_metrics( model: pl.LightningModule, in_file: str, out_file: str) -> Dict:

    chroms, seqs, labels = read_samples(in_file, out_file, keep_chrom=True)

    mx_len = max(len(seq) for seq in seqs)

    x_fw, x_rv = transform_all_sequences(seqs, mx_len)

    logits = model(x_fw, x_rv)
    preds = [1 if logit[1] >= logit[0] else 0 for logit in logits]
    labels = list(map(int, labels))

    tps, fps, tns, fns = [], [], [], []

    per_chrom = {}

    whole = {
        'tp': 0,
        'fp': 0,
        'tn': 0,
        'fn': 0,
    }

    for chrom, pred, label in zip(chroms, preds, labels):

        chrom_name = chrom.split('_')[0][1:] 
        if chrom_name not in per_chrom:
            per_chrom[chrom_name] = {
                'tp': 0,
                'fp': 0,
                'tn': 0,
                'fn': 0,
            }
        
        cur_chrom = per_chrom[chrom_name]

        if label == pred:
            if pred == 1:
                whole['tp'] += 1
                cur_chrom['tp'] += 1
                tps.append(chrom)
            else:
                whole['tn'] += 1
                cur_chrom['tn'] += 1
                tns.append(chrom)
        else:
            if pred == 1:
                whole['fp'] += 1
                cur_chrom['fp'] += 1
                fps.append(chrom)
            else:
                whole['fn'] += 1
                cur_chrom['fn'] += 1
                fns.append(chrom)

    extend(whole)
    for key in per_chrom:
        extend(per_chrom[key])

    return {
        'whole': whole,
        'chromosome-wise': per_chrom,
        'tps': tps,
        'fps': fps,
        'tns': tns,
        'fns': fns,
    }
