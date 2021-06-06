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
    rev = ''.join(inv[c] for c in reversed(seq))
    return 0 if seq.find('AG') == -1 and rev.find('AG') else 1


with open('dataset_test/raw/seq.fa', 'w') as fi, open('dataset_test/raw/out.dat', 'w') as fo:
    random.seed(0)
    pos, neg = 0, 0
    for _ in range(30000):
        seq = random_seq()
        y = label(seq)
        if y == 1 and pos > 8000:
            continue
        if y == 1: pos += 1
        else: neg += 1
        fi.write('>chr1_test_case\n')
        fi.write(seq + '\n')
        fo.write(str(y) + '\n')
    print(pos, neg)