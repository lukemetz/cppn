import tensorflow as tf
import data
from slim.ops import conv2d, avg_pool, batch_norm, fc, flatten
from ops import conv2d_transpose, lrelu
import numpy as np
from slim.scopes import arg_scope

batch_norm_params = {"epsilon":0.0001, "scale":True}
def discriminator(inp):

    n = 32
    with arg_scope([conv2d], batch_norm_params=batch_norm_params, stddev=0.02, activation=lrelu, weight_decay=1e-5):
        inp = inp-0.5
        # todo make new set of ops?
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

#def generator(z):
    #n = 32
    #with arg_scope([conv2d, conv2d_transpose], batch_norm_params=batch_norm_params, stddev=0.02, weight_decay=1e-5):
        #c = z.get_shape()[1].value / 16
        #z_in = tf.reshape(z, (-1, 4, 4, c))
        #o = conv2d_transpose(z_in, n, (3 ,3), stride=(2, 2))
        #c = z.get_shape()[1].value / (8*8)
        #z_mid = tf.reshape(z, (-1, 8, 8, c))
        #o = tf.concat(3, [o, z_mid])
        #o = conv2d_transpose(o, n, (3, 3), stride=(2, 2))
        #o = conv2d_transpose(o, n, (3, 3), stride=(2, 2))
        #o = conv2d_transpose(o, n, (3, 3), stride=(2, 2))
        ## BORKED WITH STRIDE
        #o = conv2d(o, num_filters_out=1, kernel_size=(3, 3), stride=2, padding="VALID", batch_norm_params=None)
        #out = o[:, 1:29, 1:29, :]
        #return out

def generator(z):
    n = 32
    with arg_scope([conv2d, conv2d_transpose], batch_norm_params=batch_norm_params, stddev=0.02):
        z = z*2-1
        d = 8
        z = fc(z, num_units_out=d*d*32, batch_norm_params=batch_norm_params)
        c = z.get_shape()[1].value / (d*d)
        z = tf.reshape(z, (-1, d, d, c))
        o = conv2d_transpose(z, n, (3, 3), stride=(2, 2))
        o = conv2d_transpose(o, n, (3, 3), stride=(2, 2))
        o = conv2d(o, num_filters_out=n*2, kernel_size=(3, 3), stride=1)
        o = conv2d(o, num_filters_out=1, kernel_size=(3, 3), stride=1, padding="VALID", batch_norm_params=None)
        out = o[:, 1:29, 1:29, :]
        return out


with tf.variable_scope("data"):
    images, _ = data.get_inputs(128)

with tf.variable_scope("generator") as gen_scope:
    z = tf.random_uniform([128, 16*8], 0, 1)
    generated = generator(z)
gen_scope.reuse_variables()
gen_vars = [x for x in tf.trainable_variables() if x.name.startswith(gen_scope.name)]

with tf.variable_scope("discriminator") as scope:
    real_probs = discriminator(images)

with tf.variable_scope("discriminator", reuse=True) as scope:
    fake_probs = discriminator(generated)

dis_vars = [x for x in tf.trainable_variables() if x.name.startswith(scope.name)]

discrim_loss = -(tf.log(real_probs) + tf.log(1-fake_probs))
discrim_loss_mean = tf.reduce_mean(discrim_loss)
generator_loss = -(tf.log(fake_probs))
#generator_loss = (1 - generated)**2
generator_loss_mean = tf.reduce_mean(generator_loss)


d_step = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(discrim_loss, var_list=dis_vars)
opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
g_grads_vars = opt.compute_gradients(generator_loss, var_list=gen_vars)
for g,v in g_grads_vars:
    tf.histogram_summary(g.name, g)
g_step = opt.apply_gradients(g_grads_vars)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

with tf.variable_scope(gen_scope, reuse=True):
    stable_z = np.random.uniform(0, 1, [128, 16*8]).astype("float32")
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
            _, g_loss = sess.run([g_step, generator_loss_mean])

    #writer.add_summary(sum_val, global_step=i)
    #writer.flush()
    # make and save samples
    if i % 10 == 0:
        images = sess.run(gen_images)
        grayscale_grid_vis(images[:, :, :, 0], (10, 10), save_path="out/%d.png"%i)

    print d_loss, g_loss
