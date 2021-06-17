import random

ch = 'ACGT'
inv = {
    'A': 'T',
    'C': 'G',
    'G': 'C',
    'T': 'A',
}

def random_seq() -> str:
    l = random.randint(10, 15)
    seq = ''.join(ch[random.randint(0, 3)] for _ in range(l))
    return seq
        

def label(seq: str) -> int:
    patterns = ['AGAT', 'TCTC']
    rev = ''.join(inv[c] for c in reversed(seq))
    for p in patterns:
        if seq.find(p) != -1 or rev.find(p) != -1:
            return 1
    return 0


with open('dataset_test/raw/seq.fa', 'w') as fi, open('dataset_test/raw/out.dat', 'w') as fo:
    random.seed(0)
    pos, neg = 0, 0
    for i in range(7000):
        seq = random_seq()
        y = label(seq)
        if y == 0 and neg >= 870:
            continue
        if y == 1: pos += 1
        else: neg += 1
        type = 'pos' if y == 1 else 'neg'
        fi.write(f'>chr1_{type}_{i}\n')
        fi.write(seq + '\n')
        fo.write(str(y) + '\n')
    print(pos, neg)