import matplotlib.pyplot as plt
import numpy as np


motif_base = '/home/redwan/Documents/L4-T2/Thesis/memesuite-results/cv-best-motifs/'
data_base = '/home/redwan/Documents/L4-T2/Thesis/viral-TR/globals/datasets/matrix/'
bed_base = '/home/redwan/Documents/L4-T2/Thesis/viral-TR/globals/tp_beds/matrix/'
kernel_size = 14
kernel_count = 12
version = 6
vtf = 'EBNA2-IB4'
motif_file = motif_base + f'{version}-{vtf}' + '/motif.meme'
data_file = data_base + vtf + '/raw/seq.fa'
bed_file = bed_base + vtf + f'/{version}.bed'


def read_kernels():
    kernels = []
    with open(motif_file) as f:
        for _ in range(8):
            f.readline()

        for i in range(kernel_count):
            for j in range(3):
                f.readline()
            kernels.append([list(map(float, f.readline().split())) for _ in range(kernel_size)])
    
    return kernels


def read_seqs():
    with open(bed_file) as f:
        valids = ['_'.join(line.split()[:3] + ['pos']) for line in f]

    seqs = []
    with open(data_file) as f:
        while True:
            chrom = f.readline()
            if not chrom: break
            chrom = '_'.join(chrom.split('_')[:4])[1:]
            seq = f.readline().strip()
            if chrom in valids:
                seqs.append(seq)
    return seqs


pos = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def apply_kernel(kernel, seq):
    prob_res = 0
    for i in range(len(seq) - len(kernel) + 1):
        prob_sum = 0
        for j in range(len(kernel)):
            prob_sum += kernel[j][pos[seq[i+j]]]
        prob_res = max(prob_res, prob_sum)
        # prob_res += prob_sum
    return prob_res


def make_heatmap():
    kernels = read_kernels()
    seqs = read_seqs()

    with open(f'heatmap_{vtf}_mx_tp.txt', 'w') as f:
        for seq in seqs[0:100]:
            for kernel in kernels:
                print(f'{apply_kernel(kernel, seq):.6f}', end=' ', file=f)
            print(file=f)


def draw_heatmap():
    vec = []
    with open(f'heatmap_{vtf}_mx_tp.txt') as f:
        vec = [list(map(float, line.split())) for line in f]
    vec = np.array(vec)
    plt.imshow(vec.T, cmap='hot', interpolation='nearest')
    plt.show()


make_heatmap()
draw_heatmap()