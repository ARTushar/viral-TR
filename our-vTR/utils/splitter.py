import math
import os
import random
from typing import Tuple

def write_samples(directory:str, sample_type:str, raw_in:str, raw_out:str, samples:list) -> None:
    type_in = os.path.join(directory, sample_type, raw_in)
    type_out = os.path.join(directory, sample_type, raw_out)
    with open(type_in, 'w') as fi, open(type_out, 'w') as fo:
        for (chrom, seq), label in samples:
            fi.write(chrom)
            fi.write(seq)
            fo.write(label)


def mix_merge(a:list, b:list) -> list:
    if len(a) == 0:
        return b
    if len(a) > len(b):
        a, b = b, a
    c = []
    l = len(a)
    for i in range(l):
        c.append(a[i])
        c.append(b[i])
    for i in range(l, len(b)):
        c.insert(random.randrange(0, len(c)), b[i])
    return c


def splitter(directory:str, raw_in:str, raw_out:str, test: float = .6, valid: float = .2) -> None:
    # rand = random.Random(seed)
    random.seed(0)

    pos_seqs, neg_seqs = [], []

    raw_dir_in = os.path.join(directory, 'raw', raw_in)
    raw_dir_out = os.path.join(directory, 'raw', raw_out)

    train_dir = os.path.join(directory, 'train')
    val_dir = os.path.join(directory, 'val')
    test_dir = os.path.join(directory, 'test')

    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    if not os.path.isdir(val_dir):
        os.mkdir(val_dir)
    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)

    with open(raw_dir_in, 'r') as fi:
        while True:
            chrom = fi.readline()
            if not chrom: break
            seq = fi.readline().upper()
            if 'pos' in chrom:
                pos_seqs.append(((chrom, seq), '1\n'))
            else:
                neg_seqs.append(((chrom, seq), '0\n'))

    random.shuffle(pos_seqs)
    random.shuffle(neg_seqs)

    train_seqs, val_seqs, test_seqs = [], [], []

    for l in (pos_seqs, neg_seqs):
        train_split = math.floor(test * len(l))
        val_split = math.floor(valid * len(l))

        # train_seqs = [*train_seqs, *l]

        # train_seqs = [*train_seqs, *l[0: train_split]]
        # val_seqs = [*val_seqs, *l[train_split: train_split+val_split]]
        # test_seqs = [*test_seqs, *l[train_split+val_split: -1]]
        train_seqs = mix_merge(train_seqs, l[0: train_split])
        val_seqs = mix_merge(val_seqs, l[train_split: train_split+val_split])
        test_seqs = mix_merge(test_seqs, l[train_split+val_split: -1])

    # random.shuffle(train_seqs)
    # random.shuffle(val_seqs)
    # random.shuffle(test_seqs)

    files = [
        ('train', train_seqs),
        ('val', val_seqs),
        ('test', test_seqs)
    ]

    for data_type, samples in files:
        write_samples(directory, data_type, raw_in, raw_out, samples)


def read_samples(sequence_file, label_file, keep_chrom: bool = False) -> Tuple:
    chroms = []
    sequences = []
    labels = []

    with open(sequence_file, 'r') as ifile:
        while True:
            chrom = ifile.readline()
            if not chrom: break
            seq = ifile.readline()
            if keep_chrom:
                chroms.append(chrom.strip())
            sequences.append(seq.strip())
    
    with open(label_file, 'r') as ofile:
        labels = [line.strip() for line in ofile.readlines()]

    return (chroms, sequences, labels) if keep_chrom else (sequences, labels)


def chrom_splitter(directory:str, raw_in:str, raw_out:str) -> None:
    random.seed(0)

    seqs = dict()
    labels = []

    raw_dir_in = os.path.join(directory, 'raw', raw_in)
    raw_dir_out = os.path.join(directory, 'raw', raw_out)

    train_dir = os.path.join(directory, 'train')
    val_dir = os.path.join(directory, 'val')
    test_dir = os.path.join(directory, 'test')

    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    if not os.path.isdir(val_dir):
        os.mkdir(val_dir)
    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)

    with open(raw_dir_in, 'r') as fi, open(raw_dir_out, 'r') as fo:
        while True:
            name = fi.readline()
            seq = fi.readline()
            label = fo.readline()
            if not name: break
            chrom = name.split('_')[0][1:]
            if chrom not in seqs:
                seqs[chrom] = []
            seqs[chrom].append(((name, seq), label))

    train_together, val_together, test_together = [], [], []

    for key, data in seqs.items():
        random.shuffle(data)

        train_split = math.floor(0.8 * len(data))
        val_split = math.floor(0.1 * len(data))

        train_together += data[0: train_split]
        val_together += data[train_split: train_split+val_split]
        test_together += data[train_split+val_split: -1]

    files = [
        ('train', train_together),
        ('val', val_together),
        ('test', test_together)
    ]

    for data_type, samples in files:
        write_samples(directory, data_type, raw_in, raw_out, samples)


def pos_neg_splitter(in_fas, out_pos_fa, out_neg_fa):
    with open(out_pos_fa, 'w') as fpos, open(out_neg_fa, 'w') as fneg:
        for in_fa in in_fas:
            with open(in_fa, 'r') as fi:
                while True:
                    chrom = fi.readline()
                    if not chrom: break
                    seq = fi.readline()
                    if 'pos' in chrom.strip().split('_'):
                        fpos.write(chrom)
                        fpos.write(seq)
                    else:
                        fneg.write(chrom)
                        fneg.write(seq)


if __name__ == '__main__':
    # peak_dataset = 'peak_around_datasets/normal/SRR5241430'
    # splitter('dataset1_new', 'SRR3101734_seq.fa', 'SRR3101734_out.dat')
    # chrom_splitter('dataset1_new', 'SRR3101734_seq.fa', 'SRR3101734_out.dat')
    # splitter('dataset2', 'SRR5241432_seq.fa', 'SRR5241432_out.dat')
    # splitter(peak_dataset, 'SRR5241430_seq.fa', 'SRR5241430_out.dat')
    # splitter(peak_dataset, 'SRR3101734_seq.fa', 'SRR3101734_out.dat')
    # splitter('dataset_test', 'seq.fa', 'out.dat')
    # pos_neg_splitter(
    #     ['dataset3/train/SRR5241430_seq.fa'],
    #     'gkmsvm/train_valid/pos_seq.fa',
    #     'gkmsvm/train_valid/neg_seq.fa'
    # )
    # splitter('../globals/datasets/together/normal/normal/SRR3101734',
    # 'seq.fa', 'out.dat', 0.9, 0.1)
    # splitter('../globals/datasets/normal/normal/SRR3101734', 'seq.fa', 'out.dat', 0.81, 0.09)
    splitter('../globals/datasets/matrix/EBNA2-Mutu3', 'seq.fa', 'out.dat', 0.9, 0.1)