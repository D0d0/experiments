import tensorflow as tf

i = tf.constant([1, 0, 2, 3, 0, 1, 1], dtype=tf.float32, name='i')
k = tf.constant([2, 1, 3], dtype=tf.float32, name='k')

data = tf.reshape(i, [1, int(i.shape[0]), 1], name='data')
kernel = tf.reshape(k, [int(k.shape[0]), 1, 1], name='kernel')
tf.identity(kernel, name='test_kernel')

res = tf.squeeze(tf.nn.conv1d(data, kernel, 1, 'VALID'))
tf.identity(res, name='testik')
with tf.Session() as sess:
    print(sess.run(res))

    print(sess.run(tf.get_default_graph().get_tensor_by_name('test_kernel:0')))

    f = sess.run(tf.get_default_graph().get_tensor_by_name('test_kernel:0'))
