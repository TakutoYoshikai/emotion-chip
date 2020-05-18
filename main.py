import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x = tf.placeholder(tf.float32, [1, 5])
w = tf.placeholder(tf.float32, [1, 10])
W2 = tf.Variable(tf.zeros([10, 5], tf.float32))
W1 = tf.matmul(w, W2)
y = x * W1
