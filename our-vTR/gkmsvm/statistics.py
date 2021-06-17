out_file = 'train-valid_test_cv/output.txt'

tn, tp, fn, fp = 0, 0, 0, 0

with open(out_file, 'r') as f:
    for line in f:
        chrom, value = line.split()
        if float(value) >= 0:
            if 'pos' in chrom:
                tp += 1
            else:
                fp += 1
        else:
            if 'neg' in chrom:
                tn += 1
            else:
                fn += 1

accuracy = (tp+tn)/(tp+tn+fp+fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = 2 * (precision * recall) / (precision + recall)

print('True Positives:  ', tp)
print('False Positives: ', fp)
print('True Negatives:  ', tn)
print('False Negatives: ', fn)
print('Accuracy:        ', accuracy)
print('Precision:       ', precision)
print('Recall:          ', recall)
print('F1 Score:        ', f1)