import csv

with open('pga2004.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ')
    print spamreader[1]
