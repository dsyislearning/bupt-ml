import csv

with open('result.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    count = 0
    for row in reader:
        if row[1] == 'Anomalous':
            count += 1
    print(f'Number of anomalous logs: {count}')