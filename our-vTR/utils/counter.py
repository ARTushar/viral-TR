def count(fa_file):
    pos, neg = dict(), dict()
    with open(fa_file) as f:
        for line in f:
            if line.startswith('>'):
                parts = line.split('_')
                key = parts[0][1:]
                use = pos if 'pos' in parts else neg
                if key not in use:
                    use[key] = 0
                use[key] += 1
    return pos, neg


if __name__ == '__main__':
    pos, neg = count('dataset1/raw/SRR3101734_seq.fa')
    for key in sorted(pos.keys()):
        print('{:>5}: {:>3} {:>3} {:>3} {}'.format(key, pos[key], neg[key], pos[key]-neg[key], neg[key]/pos[key]))