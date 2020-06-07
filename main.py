import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

docvec_size = 10
emovec_size = 5
x = tf.placeholder(tf.float32, [1, emovec_size])
w = tf.placeholder(tf.float32, [1, docvec_size])
Wa = tf.Variable(tf.zeros([docvec_size, emovec_size], tf.float32))
W1 = tf.matmul(w, Wa)
b1 = tf.Variable(tf.zeros([5]))
y1 = tf.sigmoid(x * W1 + b1)

W2 = tf.Variable(tf.zeros([emovec_size, emovec_size], tf.float32))
b2 = tf.Variable(tf.zeros([emovec_size]))

y2 = tf.sigmoid(y1 * W2 + b2)

y_ = tf.placeholder(tf.float32, [1, emovec_size])

y3 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=x)

sess = tf.Session()
#y = sess.run([y1, y2, y3], feed_dict={x: [], y_: []})
