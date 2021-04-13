import os

raw_in = os.path.join('raw', 'in.txt')
raw_out = os.path.join('raw', 'out.txt')
N = 1000

with open(raw_in, 'w') as fi, open(raw_out, 'w') as fo:
    for i in range(N):
        fi.write('AAAAAAAAAAAAAAAAAAAA\n');
        fo.write('1\n')
    for i in range(N):
        fi.write('TTTTTTTTTTTTTTTTTTTT\n');
        fo.write('0\n')
    for i in range(N):
        fi.write('CCCCCCCCCCCCCCCCCCCC\n');
        fo.write('1\n')
    for i in range(N):
        fi.write('GGGGGGGGGGGGGGGGGGGG\n');
        fo.write('0\n')