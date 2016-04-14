import tensorflow as tf
import data_cifar10
from slim.ops import conv2d, avg_pool, batch_norm, fc, flatten
from ops import conv2d_transpose, lrelu
import numpy as np
from slim.scopes import arg_scope
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("gpu_num", 0, "gpu number")

batch_norm_params = {"epsilon":0.0001, "scale":True}
z_dim = 16*8
z_dim = 16*8*2
z_dim = 16*8*2*8

def discrim4(inp):
  with arg_scope([conv2d], batch_norm_params=batch_norm_params, stddev=0.02, activation=lrelu, weight_decay=1e-5):
    o = conv2d(inp, num_filters_out=32, kernel_size=(3, 3), stride=1)
    o = conv2d(inp, num_filters_out=32, kernel_size=(3, 3), stride=1)
    return fc(flatten(o), num_units_out=1, activation=tf.nn.sigmoid)

def discrim8(inp):
  with arg_scope([conv2d], batch_norm_params=batch_norm_params, stddev=0.02, activation=lrelu, weight_decay=1e-5):
    o = conv2d(inp, num_filters_out=32, kernel_size=(3, 3), stride=1)
    o = conv2d(inp, num_filters_out=32, kernel_size=(3, 3), stride=2)
    o = conv2d(inp, num_filters_out=32, kernel_size=(3, 3), stride=1)
    return fc(flatten(o), num_units_out=1, activation=tf.nn.sigmoid)

def discrim16(inp):
  with arg_scope([conv2d], batch_norm_params=batch_norm_params, stddev=0.02, activation=lrelu, weight_decay=1e-5):
    o = conv2d(inp, num_filters_out=32, kernel_size=(3, 3), stride=1)
    o = conv2d(inp, num_filters_out=32, kernel_size=(3, 3), stride=2)
    o = conv2d(inp, num_filters_out=32, kernel_size=(3, 3), stride=1)
    o = conv2d(inp, num_filters_out=32, kernel_size=(3, 3), stride=2)
    o = conv2d(inp, num_filters_out=32, kernel_size=(3, 3), stride=1)
    return fc(flatten(o), num_units_out=1, activation=tf.nn.sigmoid)

def discrim32(inp):
  with arg_scope([conv2d], batch_norm_params=batch_norm_params, stddev=0.02, activation=lrelu, weight_decay=1e-5):
    o = conv2d(inp, num_filters_out=32, kernel_size=(3, 3), stride=1)
    o = conv2d(inp, num_filters_out=32, kernel_size=(3, 3), stride=2)
    o = conv2d(inp, num_filters_out=32, kernel_size=(3, 3), stride=1)
    o = conv2d(inp, num_filters_out=32, kernel_size=(3, 3), stride=2)
    o = conv2d(inp, num_filters_out=32, kernel_size=(3, 3), stride=1)
    o = conv2d(inp, num_filters_out=64, kernel_size=(3, 3), stride=2)
    o = conv2d(inp, num_filters_out=64, kernel_size=(3, 3), stride=1)
    return fc(flatten(o), num_units_out=1, activation=tf.nn.sigmoid)

def get_cords(batch_size, size):
    elem = np.mgrid[0:size, 0:size].reshape((2, -1)).T.reshape((1, -1, 2))
    elem = tf.convert_to_tensor(elem)
    res = tf.tile(elem, tf.pack([ batch_size, 1, 1]))
    res.set_shape((None, size*size, 2))
    return tf.to_float(res) / size

def triangle(x):
    sub = (.5 * x + .25)
    sub_frac = sub - tf.round(sub)
    return 1 - 4 * tf.abs(.5 - sub_frac)

def sin_bank(x, bank_size, length, scope=None):
    with tf.variable_op_scope([x], scope, "SinBank") as scope:
        with tf.device("/gpu:%d"%FLAGS.gpu_num):
            bank = tf.get_variable("bank", dtype=tf.float32, shape=[bank_size, ],
                            initializer=tf.random_uniform_initializer(0.0, length))
            shift = tf.get_variable("shift", dtype=tf.float32, shape=[bank_size, ],
                            initializer=tf.random_uniform_initializer(0.0, length))
            if not tf.get_variable_scope().reuse:
                tf.histogram_summary(bank.name, bank)
            return tf.sin(x*bank+shift)
            #return triangle(x*bank+shift)

def three_fc(x, num_units_out, *args, **kwargs):
    in_s = [y.value for y in x.get_shape()]
    flat_x = tf.reshape(x, [-1, in_s[-1]])
    o = fc(flat_x, num_units_out=num_units_out, *args, **kwargs)
    out = tf.reshape(o, [-1, in_s[1], num_units_out])
    out.set_shape( in_s[0:-1]+[num_units_out])
    return out

def cppn_func(inp, z):
    with arg_scope([fc], batch_norm_params=batch_norm_params, stddev=0.02):
        z = z*2 - 1
        #n = 32
        n = 128

        length = 20
        h = inp[:, :, 0:1]
        w = inp[:, :, 1:2]

        r_h = sin_bank(h, 64, length=length)
        fc_h = three_fc(r_h, num_units_out=n)

        r_w = sin_bank(w, 64, length=length)
        fc_w = three_fc(r_w, num_units_out=n)

        d = tf.sqrt((h-0.5)**2 + (w-0.5)**2)
        r_d = sin_bank(d, 64, length=length)
        fc_d = three_fc(r_d, num_units_out=n)

        pi = 3.1415
        n_angles = 64
        length = 20
        theta = tf.get_variable("rotations", dtype=tf.float32, shape=[n_angles,],
                        initializer=tf.random_uniform_initializer(0.0, pi*2))
        wh = tf.cos(theta) * h - tf.sin(theta)*w
        r_wh = sin_bank(wh, n_angles, length=length)
        fc_wh = three_fc(r_wh, num_units_out=n)

        length = 50
        n_angles = 64
        theta = tf.get_variable("rotations2", dtype=tf.float32, shape=[n_angles,],
                        initializer=tf.random_uniform_initializer(0.0, pi*2))

        wh_hf = tf.cos(theta) * h - tf.sin(theta)*w
        r_wh_hf = sin_bank(wh_hf, n_angles, length=length)
        fc_wh_hf = three_fc(r_wh_hf, num_units_out=n)


        length = 50
        n_angles = 128
        z_angle = fc(z, num_units_out=n_angles, activation=None, stddev=0.2)*10
        z_angle = tf.expand_dims(z_angle, 1)
        z_scale = fc(z, num_units_out=n_angles, activation=None, stddev=0.2)*30
        z_scale = tf.expand_dims(z_scale, 1)
        z_shift = fc(z, num_units_out=n_angles, activation=None, stddev=0.2)*30
        z_shift = tf.expand_dims(z_shift, 1)
        rot_z = tf.cos(z_angle) * h - tf.sin(z_angle)*w
        fc_zangle = tf.sin(rot_z*z_scale + z_shift)
        fc_zangle_proj = three_fc(fc_zangle, num_units_out=n)

        z_angle = fc(z, num_units_out=n_angles, activation=None, stddev=0.2)*10
        z_angle = tf.expand_dims(z_angle, 1)
        z_scale = fc(z, num_units_out=n_angles, activation=None, stddev=0.2)*4
        z_scale = tf.expand_dims(z_scale, 1)
        z_shift = fc(z, num_units_out=n_angles, activation=None, stddev=0.2)*4
        z_shift = tf.expand_dims(z_shift, 1)
        rot_z = tf.cos(z_angle) * h - tf.sin(z_angle)*w
        fc_zangle = tf.sin(rot_z*z_scale + z_shift)
        fc_zangle_proj_large = three_fc(fc_zangle, num_units_out=n)


        z_comb = fc(z, num_units_out=n)
        z_comb = tf.expand_dims(z_comb, 1)

        #res = (fc_h + fc_w + fc_d) * context_proc + z_comb
        #res = (fc_h + fc_w + fc_d + fc_wh) + z_comb
        #res = (fc_wh + fc_wh_hf) + z_comb
        #res = (fc_wh + fc_wh_hf + fc_d + fc_zangle_proj) + z_comb
        #res = (fc_zangle_proj + fc_zangle_proj_large) + z_comb
        res = (fc_wh + fc_wh_hf + fc_d + fc_zangle_proj + fc_zangle_proj_large) + z_comb
        #res = (fc_h + fc_w + fc_d) + z_comb
        #res = (fc_h + fc_w + fc_d) + z_comb
        #res = fc_h + fc_w
        z_mul = fc(z, num_units_out=n)
        z_mul = tf.expand_dims(z_mul, 1)

        #res *= z_mul

        n = 64
        h = three_fc(res, num_units_out=n)
        h2 = three_fc(h, num_units_out=n)
        #h3 = three_fc(h2, num_units_out=n)
        return three_fc(h2, num_units_out=3, batch_norm_params=None)

def generator(z, size=32):
    with tf.device("/gpu:%d"%FLAGS.gpu_num):
        coords = get_cords(tf.shape(z)[0], size=size)
        # coords: batch x size*size x 2
        cppn_result_flat = cppn_func(coords,z)
        result_image = tf.reshape(cppn_result_flat, [-1, size, size, 3])
        return result_image

#def generator(z, size=32):
    #with arg_scope([conv2d, conv2d_transpose], batch_norm_params=batch_norm_params, stddev=0.02):
        #with tf.device("/gpu:%d"%FLAGS.gpu_num):
            #z = z*2-1
            #d = 4
            #z = fc(z, num_units_out=d*d*32, batch_norm_params=batch_norm_params)
            #c = z.get_shape()[1].value / (d*d)
            #z = tf.reshape(z, (-1, d, d, c))
            #n = 128
            #o = conv2d_transpose(z, n, (3, 3), stride=(2, 2))
            #o = conv2d_transpose(o, n, (3, 3), stride=(2, 2))
            #o = conv2d(o, num_filters_out=n, kernel_size=(3, 3), stride=1)
            #n = 64
            #n = 128
            #o = conv2d_transpose(o, n, (3, 3), stride=(2, 2))t2
            #o = conv2d(o, num_filters_out=n, kernel_size=(3, 3), stride=1)
            #o = conv2d(o, num_filters_out=3, kernel_size=(3, 3), stride=1)
            #attended = o
            #return attended


#batch_size = 32
#batch_size = 64
batch_size = 48

with tf.variable_scope("data"):
    with tf.device("/cpu:0"):
        images32, _ = data_cifar10.get_inputs(batch_size)

        images16, _ = data_cifar10.get_inputs(batch_size)
        images16 = tf.image.resize_images(images16, 16, 16)

        images8, _ = data_cifar10.get_inputs(batch_size)
        images8 = tf.image.resize_images(images16, 8, 8)

        images4, _ = data_cifar10.get_inputs(batch_size)
        images4 = tf.image.resize_images(images16, 4, 4)

with tf.variable_scope("generator") as gen_scope:
    z = tf.random_uniform([batch_size, z_dim], 0, 1)
    gen4 = generator(z, size=4)

print [n.name for n in tf.trainable_variables()]
with tf.variable_scope(gen_scope, reuse=True):
    z = tf.random_uniform([batch_size, z_dim], 0, 1)
    gen8 = generator(z, size=8)

with tf.variable_scope(gen_scope, reuse=True):
    z = tf.random_uniform([batch_size, z_dim], 0, 1)
    gen16 = generator(z, size=16)

with tf.variable_scope(gen_scope, reuse=True):
    z = tf.random_uniform([batch_size, z_dim], 0, 1)
    gen32 = generator(z, size=32)

gen_vars = [x for x in tf.trainable_variables() if x.name.startswith(gen_scope.name)]

with tf.variable_scope("discriminator") as scope:
    real_probs4 = discrim4(images4)
    real_probs8 = discrim8(images8)
    real_probs16 = discrim16(images16)
    real_probs32 = discrim32(images32)

with tf.variable_scope("discriminator", reuse=True) as scope:
    fake_probs4 = discrim4(gen4)
    fake_probs8 = discrim8(gen8)
    fake_probs16 = discrim16(gen16)
    fake_probs32 = discrim32(gen32)

dis_vars = [x for x in tf.trainable_variables() if x.name.startswith(scope.name)]

def discrim_gen_loss(real_probs, fake_probs):
    discrim_loss = -(tf.log(real_probs) + tf.log(1-fake_probs))
    discrim_loss_mean = tf.reduce_mean(discrim_loss)
    generator_loss = -(tf.log(fake_probs))
    generator_loss_mean = tf.reduce_mean(generator_loss)
    return discrim_loss, discrim_loss_mean, generator_loss, generator_loss_mean

d4_loss, d4_mean_loss, g4_loss, g4_mean_loss = discrim_gen_loss(real_probs4, fake_probs4)
d8_loss, d8_mean_loss, g8_loss, g8_mean_loss = discrim_gen_loss(real_probs8, fake_probs8)
d16_loss, d16_mean_loss, g16_loss, g16_mean_loss = discrim_gen_loss(real_probs16, fake_probs16)
d32_loss, d32_mean_loss, g32_loss, g32_mean_loss = discrim_gen_loss(real_probs32, fake_probs32)

d4_step = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(d4_loss, var_list=dis_vars)
d8_step = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(d8_loss, var_list=dis_vars)
d16_step = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(d16_loss, var_list=dis_vars)
d32_step = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(d32_loss, var_list=dis_vars)

#opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
g4_step= tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(g4_loss, var_list=gen_vars)
g8_step= tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(g8_loss, var_list=gen_vars)
g16_step= tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(g16_loss, var_list=gen_vars)
g32_step= tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(g32_loss, var_list=gen_vars)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

with tf.variable_scope(gen_scope, reuse=True):
    stable_z = np.random.uniform(0, 1, [7*7, z_dim]).astype("float32")
    tensor_z = tf.convert_to_tensor(stable_z)
    gen_images = generator(tensor_z, size=32)

with tf.variable_scope(gen_scope, reuse=True):
    gen_images_4x = generator(tensor_z, size=32*4)

summary = tf.merge_all_summaries()
init = tf.initialize_all_variables()
sess.run(init)
tf.train.start_queue_runners(sess)

writer = tf.train.SummaryWriter("logs/", graph=sess.graph)

from scipy.misc import imsave
import skimage.io
def color_grid_vis(X, (nh, nw), save_path=None):
    h, w = X[0].shape[0:2]
    img = np.zeros((h*nh, w*nw, 3))
    for n, x in enumerate(X[0:h*w]):
        if n == nh*nw:
            break
        j = n/nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w, :] = x
    if save_path is not None:
        img = (np.clip(img, 0, 1)*255).astype("uint8")
        skimage.io.imsave(save_path, img)
    return img

i = 0

import sys
import json
if len(sys.argv) == 2:
    ff = open("logs/%s.ndjson"%sys.argv[1], "w")

while True:
    print "<", i , ">"
    _, d4_loss = sess.run([d4_step, d4_mean_loss])
    _, g4_loss = sess.run([g4_step, g4_mean_loss])

    _, d8_loss = sess.run([d8_step, d8_mean_loss])
    _, g8_loss = sess.run([g8_step, g8_mean_loss])

    _, d16_loss = sess.run([d16_step, d16_mean_loss])
    _, g16_loss = sess.run([g16_step, g16_mean_loss])

    _, d32_loss = sess.run([d32_step, d32_mean_loss])
    _, g32_loss = sess.run([g32_step, g32_mean_loss])

    #sum_val, _, ae_l, kl_l = sess.run([ summary, ae_step, ae_loss_mean, kl_loss_mean])
    #print ae_l, kl_l
    print "4 > dloss:", d4_loss, "gloss:", g4_loss
    print "8 > dloss:", d8_loss, "gloss:", g8_loss
    print "16 > dloss:", d16_loss, "gloss:", g16_loss
    print "32 > dloss:", d32_loss, "gloss:", g32_loss


    if i % 50 == 0:
        images = sess.run(gen_images)
        images_4x = sess.run(gen_images_4x)
        color_grid_vis(images[:, :, :, :], (7, 7), save_path="out2/%d.png"%i)
        color_grid_vis(images_4x[:, :, :, :], (7, 7), save_path="out2/%d_4x.png"%i)

    i += 1

