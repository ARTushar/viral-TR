import math
import os
import random
from typing import Tuple

from torch.functional import split


def write_samples(directory:str, sample_type:str, raw_in:str, raw_out:str, samples:list) -> None:
    type_in = os.path.join(directory, sample_type, raw_in)
    type_out = os.path.join(directory, sample_type, raw_out)
    with open(type_in, 'w') as fi, open(type_out, 'w') as fo:
        for (chrom, seq), label in samples:
            fi.write(chrom)
            fi.write(seq)
            fo.write(label)


def splitter(directory:str, raw_in:str, raw_out:str, test: float = .6, valid: float = .2) -> None:
    # rand = random.Random(seed)

    all_seqs = []

    raw_dir_in = os.path.join(directory, 'raw', raw_in)
    raw_dir_out = os.path.join(directory, 'raw', raw_out)

    train_dir = os.path.join(directory, 'train')
    cv_dir = os.path.join(directory, 'cv')
    test_dir = os.path.join(directory, 'test')

    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    if not os.path.isdir(cv_dir):
        os.mkdir(cv_dir)
    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)

    with open(raw_dir_in, 'r') as fi:
        while True:
            chrom = fi.readline()
            if not chrom: break
            seq = fi.readline().upper()
            all_seqs.append((chrom, seq))

    all_labels = []
    with open(raw_dir_out, 'r') as fo:
        all_labels = list(fo.readlines())

    all_together = list(zip(all_seqs, all_labels))
    random.shuffle(all_together)

    train_split = math.floor(test * len(all_together))
    cv_split = math.floor(valid * len(all_together))

    train_together = all_together[0: train_split]
    cv_together = all_together[train_split: train_split+cv_split]
    test_together = all_together[train_split+cv_split: -1]

    files = [
        ('train', train_together),
        ('cv', cv_together),
        ('test', test_together)
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
    cv_dir = os.path.join(directory, 'cv')
    test_dir = os.path.join(directory, 'test')

    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    if not os.path.isdir(cv_dir):
        os.mkdir(cv_dir)
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
        ('cv', val_together),
        ('test', test_together)
    ]

    for data_type, samples in files:
        write_samples(directory, data_type, raw_in, raw_out, samples)


if __name__ == '__main__':
    peak_dataset = 'peak_around_datasets/chrom_wise/SRR5241430'
    # splitter('dataset1_new', 'SRR3101734_seq.fa', 'SRR3101734_out.dat')
    # chrom_splitter('dataset1_new', 'SRR3101734_seq.fa', 'SRR3101734_out.dat')
    # splitter('dataset2', 'SRR5241432_seq.fa', 'SRR5241432_out.dat')
    splitter(peak_dataset, 'SRR5241430_seq.fa', 'SRR5241430_out.dat')
    # splitter(peak_dataset, 'SRR3101734_seq.fa', 'SRR3101734_out.dat')
    # splitter('dataset_test', 'seq.fa', 'out.dat')