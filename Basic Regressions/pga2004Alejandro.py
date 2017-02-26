# Dataset:   pga2004.dat

# Source: sportsillustrated.cnn.com

# Description: Performance statistics and winnings for 196 PGA participants
# during 2004 season.

import tensorflow as tf
import csv

with open('dos_datos.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        print(row['x'])