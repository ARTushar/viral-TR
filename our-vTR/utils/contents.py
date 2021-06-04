import json

def content(seq_file):
    with open(seq_file) as f:
        avg = {'A': 0.0, 'C': 0.0, 'G': 0.0, 'T': 0.0}
        total = 0
        for line in f:
            cnt = {'A': 0.0, 'C': 0.0, 'G': 0.0, 'T': 0.0}
            if not line.startswith('>'):
                line = line.strip()
                for ch in line:
                    cnt[ch] += 1
                l = len(line)
                for key in cnt:
                    cnt[key] /= l
                    avg[key] += cnt[key]
                total += 1
        for key in avg:
            avg[key] /= total
        print(json.dumps(avg, indent=4))


# content('dataset1_dummy/raw/SRR3101734_seq.fa')
content('dataset3/raw/SRR5241430_seq.fa')
