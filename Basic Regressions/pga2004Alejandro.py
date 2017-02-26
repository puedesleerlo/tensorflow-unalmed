# Dataset:   pga2004.dat

# Source: sportsillustrated.cnn.com

# Description: Performance statistics and winnings for 196 PGA participants
# during 2004 season.

import tensorflow as tf
import csv
import numpy
def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)
with open('dos_datos.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    independent = []
    dependent = []
    number = 0
    for row in reader:
        x = num(row['x'])
        y = num(row['y'])
        independent.append(x)
        dependent.append(y)

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Training Data
train_X = numpy.asarray(independent)
train_Y = numpy.asarray(dependent)
n_samples = train_X.shape[0]

X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
pred = tf.add(tf.mul(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()