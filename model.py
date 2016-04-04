import tensorflow as tf
import data
from matplotlib import pylab as plt
import numpy as np
from tqdm import trange

imgs, labels = data.get_inputs(batch_size=10)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def sample_img(img, n_samples):
    sx = tf.random_uniform((n_samples,), 0, 1) * 27
    sy = tf.random_uniform((n_samples,), 0, 1) * 27
    sx_lower = tf.cast(tf.floor(sx), tf.int32)
    sx_upper = tf.cast(tf.ceil(sx), tf.int32)

    sy_lower = tf.cast(tf.floor(sy), tf.int32)
    sy_upper = tf.cast(tf.ceil(sy), tf.int32)

    sx_nearest = tf.cast(tf.round(sx), tf.int32)
    sy_nearest = tf.cast(tf.round(sy), tf.int32)
    inds = tf.pack([sx_nearest, sy_nearest])
    samples = tf.gather(tf.reshape(img, (-1,)), sx_nearest + sy_nearest*28)
    return sx/27, sy/27, samples

samp_x, samp_y, samples = sample_img(imgs[0, :, :], 2048)

def model(samp_x, samp_y):
    samp_x = tf.expand_dims(samp_x, -1)
    samp_y = tf.expand_dims(samp_y, -1)
    n = 512
    o1 = tf.nn.relu(linear(samp_x, output_size=n, scope="l1x"))
    o2 = tf.nn.relu(linear(samp_y, output_size=n, scope="l1y"))
    o = o1 + o2
    o = tf.nn.relu((linear(o, output_size=n, scope="l2")))
    o = tf.nn.relu((linear(o, output_size=n, scope="l2s")))
    o = tf.nn.relu((linear(o, output_size=n, scope="l4s")))
    o = tf.nn.sigmoid(linear(o, output_size=1, scope="l3"))
    return tf.squeeze(o, [1])

def mse(pred, target):
    return tf.square(target-pred)

def reconstruct_samples(target_size):
    idxs = np.mgrid[0:target_size,0:target_size].reshape(2, -1)
    sx = idxs[0]/float(target_size)
    sy = idxs[1]/float(target_size)
    return sy, sx

def reconstruct_image(res, target_size):
    return res.reshape((target_size, target_size))

#tf.get_variable_scope().reuse_variables()

p_sx = tf.placeholder(shape=[None,], dtype=tf.float32)
p_sy = tf.placeholder(shape=[None,], dtype=tf.float32)
p_res = tf.placeholder(shape=[None,], dtype=tf.float32)
res = model(p_sx, p_sy)

loss = mse(res, p_res)
mean_loss = tf.reduce_mean(loss)
train_op = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(loss)


sx, sy = reconstruct_samples(128)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run([init])
tf.train.start_queue_runners(sess=sess)
x, y, saps = sess.run([samp_x, samp_y, samples])
for i in trange(3000):
    _, ll = sess.run([train_op, mean_loss], feed_dict={p_sx:x, p_sy:y, p_res:saps})
    print ll

res_val = sess.run(res, feed_dict={p_sx:sx.astype("float32"), p_sy:sy.astype("float32")})
img = reconstruct_image(res_val, 128)
plt.imshow(img, cmap="gray", interpolation="nearest")
plt.show()

plt.scatter(x, y, c=saps, lw=0)
plt.show()

import ipdb; ipdb.set_trace()
    # pull a triangle filter

# pull a bunch of samples from an image
