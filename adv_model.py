import tensorflow as tf
import data
from slim.ops import conv2d, avg_pool, batch_norm, lrelu


def discriminator(inp):
    # todo make new set of ops?
    o = conv2d(inp, kernel_size=(3,3), stride=2)

images, _ = data.get_inputs(128)

z = tf.random.uniform([128], 0, 1)

generated = generate_image(z, target_size=32)

with tf.variable_scope("discriminator") as scope:
    real_probs = discriminator(images)
    scope.reuse_variables()
    fake_probs = discriminator(generated)

discrim_loss = tf.log(real_probs) + tf.log(1-fake_probs)
generator_loss = tf.log(fake_probs)
