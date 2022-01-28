import matplotlib.pyplot as plt
import numpy as np


motif_base = '/home/redwan/Documents/L4-T2/Thesis/memesuite-results/cv-best-motifs/'
data_base = '/home/redwan/Documents/L4-T2/Thesis/viral-TR/globals/datasets/matrix/'
kernel_size = 14
kernel_count = 16
version = 18
vtf = 'HBZ-ST1'
motif_file = motif_base + f'{version}-{vtf}' + '/motif.meme'
data_file = data_base + vtf + '/raw/seq.fa'


def read_kernels(input_file):
    kernels = []
    with open(input_file) as f:
        for _ in range(8):
            f.readline()

        for i in range(kernel_count):
            for j in range(3):
                f.readline()
            kernels.append([list(map(float, f.readline().split())) for _ in range(kernel_size)])
    
    return kernels


def read_seqs(input_file):
    seqs = []
    with open(input_file) as f:
        while True:
            chrom = f.readline()
            if not chrom: break
            seq = f.readline().strip()
            seqs.append(seq)
    return seqs


def get_true_pos_seqs(seqs):
    model = 


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
    kernels = read_kernels(motif_file)
    seqs = read_seqs(data_file)

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


# make_heatmap()
draw_heatmap()