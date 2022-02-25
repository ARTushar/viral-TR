import os
import torch


def get_tp_chroms(logits, y, chroms):
    y_pred = torch.round(logits[:, 1])
    y = torch.argmax(y, 1)
    unique = torch.Tensor([x + 2*y for x, y in zip(y_pred, y)])
    tp = torch.where(unique == 3)[0].tolist()
    return [chroms[i] for i in tp]


def tp_chroms_to_bed_file(chroms, dataset_dir, version):
    print('TP Bed File for ', dataset_dir, ' version: ', version)
    GDIR = os.path.join('..', 'globals')
    tp_dir = os.path.join(GDIR, 'tp_beds', dataset_dir)
    if not os.path.isdir(tp_dir):
        os.makedirs(tp_dir)
    
    with open(os.path.join(tp_dir, str(version) + '.bed'), 'w') as file:
        for chrom in chroms:
            chrom = chrom[1:]
            c = chrom.split('_')
            file.write(c[0])
            file.write('\t')
            file.write(c[1])
            file.write('\t')
            file.write(c[2])
            file.write('\t')
            # file.write('.')
            # file.write('\t')
            # file.write(c[4])
            # file.write('\t')
            file.write('.\n')
