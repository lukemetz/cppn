from cifar10_input import distorted_inputs
import tensorflow as tf

inputs, labels = distorted_inputs("/home/luke/datasets/cifar10/cifar-10-batches-bin", 128)
for x in range(10):
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run([init])
    dat = sess.run([inputs])

import ipdb; ipdb.set_trace()
