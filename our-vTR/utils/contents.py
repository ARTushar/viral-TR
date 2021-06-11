import json

def content(seq_file):
    with open(seq_file) as f:
        pos_avg = {'A': 0.0, 'C': 0.0, 'G': 0.0, 'T': 0.0}
        neg_avg = {'A': 0.0, 'C': 0.0, 'G': 0.0, 'T': 0.0}
        pos_total, neg_total = 0, 0
        while True:
            chrom = f.readline()
            if not chrom: break
            seq = f.readline()
            cnt = {'A': 0.0, 'C': 0.0, 'G': 0.0, 'T': 0.0}
            seq = seq.strip()
            for ch in seq:
                cnt[ch] += 1
            l = len(seq)
            if 'pos' in chrom.split('_'):
                for key in cnt:
                    pos_avg[key] += cnt[key] / l
                pos_total += 1
            else:
                for key in cnt:
                    neg_avg[key] += cnt[key] / l
                neg_total += 1

        print(pos_total, neg_total)

        for key in pos_avg:
            pos_avg[key] /= pos_total
            neg_avg[key] /= neg_total
        print(json.dumps(pos_avg, indent=4))
        print(json.dumps(neg_avg, indent=4))


content('dataset1_dummy/raw/SRR3101734_seq.fa')
# content('dataset3/raw/SRR5241430_seq.fa')
