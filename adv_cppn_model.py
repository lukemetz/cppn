import tensorflow as tf
import data
from slim.ops import conv2d, avg_pool, batch_norm, fc, flatten
from ops import conv2d_transpose, lrelu
import numpy as np
from slim.scopes import arg_scope

batch_norm_params = {"epsilon":0.0001, "scale":True}
z_dim = 16*8

def discriminator(inp):
    n = 32
    with arg_scope([conv2d], batch_norm_params=batch_norm_params, stddev=0.02, activation=lrelu, weight_decay=1e-5):
        inp = inp-0.5
        o = conv2d(inp, num_filters_out=n, kernel_size=(3, 3), stride=1)
        o = conv2d(o, num_filters_out=n, kernel_size=(3, 3), stride=2)
        #o = conv2d(o, num_filters_out=n, kernel_size=(3, 3), stride=1)
        o = conv2d(o, num_filters_out=n, kernel_size=(3, 3), stride=2)
        o = conv2d(o, num_filters_out=n, kernel_size=(3, 3), stride=2)
        o = conv2d(o, num_filters_out=n, kernel_size=(3, 3), stride=1)
        flat = flatten(o)
        #flat = flatten(avg_pool(o, kernel_size=3))
        prob = fc(flat, num_units_out=1, activation=tf.nn.sigmoid)
        #prob = tf.Print(prob, [prob])
        return prob

def generator_context(z):
    n = 32
    with arg_scope([conv2d, conv2d_transpose], batch_norm_params=batch_norm_params, stddev=0.02):
        z = z*2-1
        d = 8
        z = fc(z, num_units_out=d*d*32, batch_norm_params=batch_norm_params)
        c = z.get_shape()[1].value / (d*d)
        z = tf.reshape(z, (-1, d, d, c))
        o = conv2d_transpose(z, n, (3, 3), stride=(2, 2))
        o = conv2d_transpose(o, n, (3, 3), stride=(2, 2))
        o = conv2d(o, num_filters_out=n, kernel_size=(3, 3), stride=1)
        o = conv2d(o, num_filters_out=4, kernel_size=(3, 3), stride=1)
        attended = o
        return attended

def get_cords(batch_size):
    elem = np.mgrid[0:28, 0:28].reshape((2, -1)).T.reshape((1, -1, 2))
    elem = tf.convert_to_tensor(elem)
    res = tf.tile(elem, tf.pack([ batch_size, 1, 1]))
    res.set_shape((None, 28*28, 2))
    return tf.to_float(res) / 28.

def sin_bank(x, bank_size, scope=None):
    with tf.variable_op_scope([x], scope, "SinBank") as scope:
        bank = tf.get_variable("bank", dtype=tf.float32, shape=[bank_size, ],
                        initializer=tf.random_uniform_initializer(0.0, 20.0))
        shift = tf.get_variable("shift", dtype=tf.float32, shape=[bank_size, ],
                        initializer=tf.random_uniform_initializer(0.0, 20.0))
        if not tf.get_variable_scope().reuse:
            tf.histogram_summary(bank.name, bank)
        return tf.sin(x*bank+shift)

def three_fc(x, num_units_out, *args, **kwargs):
    in_s = [y.value for y in x.get_shape()]
    flat_x = tf.reshape(x, [-1, in_s[-1]])
    o = fc(flat_x, num_units_out=num_units_out, *args, **kwargs)
    out = tf.reshape(o, [-1, in_s[1], num_units_out])
    out.set_shape( in_s[0:-1]+[num_units_out])
    return out

def cppn_func(inp, context):
    with arg_scope([fc], batch_norm_params=batch_norm_params, stddev=0.02):
        n = 64
        h = inp[:, :, 0:1]
        w = inp[:, :, 1:2]
        d = tf.sqrt((h-0.5)**2 + (w-0.5)**2)

        r_h = sin_bank(h, 128)
        fc_h = three_fc(r_h, num_units_out=n)

        r_w = sin_bank(w, 128)
        fc_w = three_fc(r_w, num_units_out=n)

        r_d = sin_bank(d, 128)
        fc_d = three_fc(r_d, num_units_out=n)


        context_proc = fc(flatten(context), num_units_out=n)
        context_proc = tf.expand_dims(context_proc, 1)

        res = (fc_h + fc_w + fc_d) * context_proc
        #res = fc_h + fc_w
        hidden = three_fc(res, num_units_out=n)
        return three_fc(hidden, num_units_out=1, batch_norm_params=None)

def generator(z):
    attended = generator_context(z)
    coords = get_cords(tf.shape(z)[0])
    # coords: batch x 28*28 x 2
    cppn_result_flat = cppn_func(coords, attended)
    result_image = tf.reshape(cppn_result_flat, [-1, 28, 28, 1])
    return result_image

def encoder(inp, z_dim):
    n = 32
    with arg_scope([conv2d], batch_norm_params=batch_norm_params, stddev=0.02, activation=lrelu, weight_decay=1e-5):
        inp = inp-0.5
        o = conv2d(inp, num_filters_out=n, kernel_size=(3, 3), stride=1)
        o = conv2d(o, num_filters_out=n, kernel_size=(3, 3), stride=2)
        o = conv2d(o, num_filters_out=n, kernel_size=(3, 3), stride=2)
        o = conv2d(o, num_filters_out=n, kernel_size=(3, 3), stride=2)
        o = conv2d(o, num_filters_out=n, kernel_size=(3, 3), stride=1)
        flat = flatten(o)
        #flat = flatten(avg_pool(o, kernel_size=3))
        z = fc(flat, num_units_out=z_dim, activation=tf.nn.tanh)/2+.5
        return z

batch_size = 32

with tf.variable_scope("data"):
    images, _ = data.get_inputs(batch_size)

with tf.variable_scope("generator") as gen_scope:
    z = tf.random_uniform([batch_size, z_dim], 0, 1)
    generated = generator(z)
gen_scope.reuse_variables()
gen_vars = [x for x in tf.trainable_variables() if x.name.startswith(gen_scope.name)]

with tf.variable_scope("discriminator") as scope:
    real_probs = discriminator(images)

with tf.variable_scope("discriminator", reuse=True) as scope:
    fake_probs = discriminator(generated)

with tf.variable_scope("encoder") as encoder_scope:
    enc_z = encoder(images, z_dim)
    enc_z_mean = fc(enc_z, num_units_out=z_dim, batch_norm_params=None, activation=None)
    enc_z_sigma = fc(enc_z, num_units_out=z_dim, batch_norm_params=None, activation=None)
    zz = tf.shape(enc_z)
    enc_sampled_z = tf.exp(enc_z_sigma) * tf.random_normal(tf.shape(enc_z)) + enc_z_mean

with tf.variable_scope(gen_scope, reuse=True):
    ae_generated = generator(enc_sampled_z)

dis_vars = [x for x in tf.trainable_variables() if x.name.startswith(scope.name)]


discrim_loss = -(tf.log(real_probs) + tf.log(1-fake_probs))
discrim_loss_mean = tf.reduce_mean(discrim_loss)
generator_loss = -(tf.log(fake_probs))
#generator_loss = (1 - generated)**2
generator_loss_mean = tf.reduce_mean(generator_loss)

mse_loss = tf.reduce_mean((images - ae_generated)**2, reduction_indices=[1, 2, 3])
kl_loss = -0.5 * (1 + 2*enc_z_sigma - enc_z_mean**2 - tf.exp(2*enc_z_sigma))
kl_loss = tf.reduce_sum(kl_loss, reduction_indices=[1])
kl_loss_mean = tf.reduce_mean(kl_loss)
ae_loss = mse_loss + kl_loss
ae_loss_mean = tf.reduce_mean(ae_loss)

d_step = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(discrim_loss, var_list=dis_vars)
opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
g_grads_vars = opt.compute_gradients(generator_loss, var_list=gen_vars)

ae_step = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(ae_loss)
#for g,v in g_grads_vars:
    #tf.histogram_summary(g.name, g)
g_step = opt.apply_gradients(g_grads_vars)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

with tf.variable_scope(gen_scope, reuse=True):
    stable_z = np.random.uniform(0, 1, [128, z_dim]).astype("float32")
    gen_images = generator(tf.convert_to_tensor(stable_z))

summary = tf.merge_all_summaries()
init = tf.initialize_all_variables()
sess.run(init)
tf.train.start_queue_runners(sess)

writer = tf.train.SummaryWriter("logs/", graph=sess.graph)

from scipy.misc import imsave
import skimage.io
def grayscale_grid_vis(X, (nh, nw), save_path=None):
    h, w = X[0].shape[-2:]
    img = np.zeros((h*nh, w*nw))
    for n, x in enumerate(X[0:h*w]):
        if n == nh*nw:
            break
        j = n/nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w] = x
    if save_path is not None:
        img = (np.clip(img, 0, 1)*255).astype("uint8")
        skimage.io.imsave(save_path, img)
    return img

i = 0
for i in range(10):
    _, d_loss = sess.run([d_step, discrim_loss_mean])
while True:
    i += 1
    _, d_loss = sess.run([d_step, discrim_loss_mean])

    _, g_loss = sess.run([g_step, generator_loss_mean])

    if g_loss > 1:
        for j in range(int(g_loss)):
            #_, ae_l = sess.run([ae_step, ae_loss_mean])
            _, g_loss = sess.run([g_step, generator_loss_mean])
            _, g_loss = sess.run([g_step, generator_loss_mean])

    #sum_val, _, ae_l, kl_l = sess.run([ summary, ae_step, ae_loss_mean, kl_loss_mean])
    print d_loss, g_loss
    #print ae_l, kl_l

    #writer.add_summary(sum_val, global_step=i)
    #writer.flush()
    # make and save samples
    if i % 10 == 0:
        images = sess.run(gen_images)
        grayscale_grid_vis(images[:, :, :, 0], (10, 10), save_path="out/%d.png"%i)

