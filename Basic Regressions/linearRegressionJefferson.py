import tensorflow as tf

a = tf.constant(5)
b = tf.constant(4)

suma = tf.add(a,b)
mul = tf.matmul(a,b)

print (mul)
print (suma)
