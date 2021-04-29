import csv

with open('params_log_train.csv', 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=d.keys())
    writer.writerow(d)