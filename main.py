import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x = tf.placeholder(tf.float32, [1, 5])
w = tf.placeholder(tf.float32, [1, 10])
W2 = tf.Variable(tf.zeros([10, 5], tf.float32))
W1 = tf.matmul(w, W2)
b = tf.Variable(tf.zeros([5]))
y = tf.sigmoid(x * W1 + b)
y_ = tf.placeholder(tf.float32, [1, 5])
cross_entropy = tf.reduce_mean(tf.square(y - y_))
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
