import pandas as pd


pd.options.display.float_format = '  {:.2e}'.format


def main():
    file_base = '/home/redwan/Documents/L4-T2/Thesis/memesuite-results/cv-best-motifs/results/core-vs-individual/'

    celllines = ['EBNA2-Mutu3', 'EBNA2-IB4', 'EBNA2-GM12878']
    # celllines = ['EBNA2-Mutu3', 'EBNA3ABC-Mutu3']
    # celllines = ['EBNA2-IB4', 'EBNALP-IB4']

    n = len(celllines)

    results, motifs = [], []
    for cellline in celllines:
        tomtom_file = file_base + cellline + '/tomtom.tsv'
        result = pd.read_csv(tomtom_file, sep='\t')[:-3]
        # print('{} headers: {}'.format(cellline, result.columns.values))
        motif = result[['Target_ID', 'E-value']]
        motif = motif.sort_values(by=['E-value']).drop_duplicates(subset=['Target_ID'])
        # print('{} motifs:'.format(cellline))
        # print(motif)
        # print()
        results.append(result)
        motifs.append(motif)
    
    # pairwise intersection
    for i in range(n):
        for j in range(i+1, n):
            intersection = motifs[i].merge(motifs[j], how='inner', on=['Target_ID'])
            print('{} intersect {}'.format(celllines[i], celllines[j]))
            print(intersection)
            print()

    if n > 2:
        # all intersection
        intersection = motifs[0]
        for i in range(1, n):
            intersection = intersection.merge(motifs[i], how='inner', on=['Target_ID'])
        print('all {} intersect'.format(n))
        print(intersection)
        print()

    # pairwise set difference
    for i in range(n):
        for j in range(n):
            if i != j:
                i_minus_j = pd.concat([motifs[i], motifs[j], motifs[j]]).drop_duplicates(subset=['Target_ID'], keep=False)
                print('{} minus {}'.format(celllines[i], celllines[j]))
                print(i_minus_j)
                print()

    if n > 2:
        # one minus all set difference
        for i in range(n):
            motif = motifs[i]
            for j in range(n):
                if i != j:
                    motif = pd.concat([motif, motifs[j], motifs[j]]).drop_duplicates(subset=['Target_ID'], keep=False)
            print('{} minus all'.format(celllines[i]))
            print(motif)
            print()


if __name__ == '__main__':
    main()