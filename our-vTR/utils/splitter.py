import math
import os
import random


def write_samples(directory:str, sample_type:str, raw_in:str, raw_out:str, samples:list) -> None:
    type_in = os.path.join(directory, sample_type, raw_in)
    type_out = os.path.join(directory, sample_type, raw_out)
    with open(type_in, 'w') as fi, open(type_out, 'w') as fo:
        for sample in samples:
            fi.write(sample[0])
            fo.write(sample[1])


def splitter(directory:str, raw_in:str, raw_out:str) -> None:
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
        for line in fi:
            line = line.upper()
            if line[0] != '>':
                all_seqs.append(line)

    all_labels = []
    with open(raw_dir_out, 'r') as fo:
        all_labels = list(fo.readlines())

    all_together = list(zip(all_seqs, all_labels))
    random.shuffle(all_together)

    train_split = math.floor(0.9 * len(all_together))
    cv_split = math.floor(0.02 * len(all_together))

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


def read_samples(sequence_file, label_file):
    sequences = []
    labels = []

    with open(sequence_file, 'r') as ifile:
        sequences = [line.strip() for line in ifile.readlines()]
    
    with open(label_file, 'r') as ofile:
        labels = [line.strip() for line in ofile.readlines()]

    return sequences, labels