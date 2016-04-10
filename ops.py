import slim.scopes as scopes
import slim.variables as variables
import tensorflow as tf
from slim.ops import _two_element_tuple, batch_norm

@scopes.add_arg_scope
def conv2d_transpose(inputs,
           num_filters_out,
           kernel_size,
           stride=1,
           padding='SAME',
           activation=tf.nn.relu,
           stddev=0.01,
           bias=0.0,
           weight_decay=0,
           batch_norm_params=None,
           is_training=True,
           trainable=True,
           restore=True,
           scope=None,
           reuse=None):

  with tf.variable_op_scope([inputs], scope, 'Conv_Transpose', reuse=reuse):
    kernel_h, kernel_w = _two_element_tuple(kernel_size)
    stride_h, stride_w = _two_element_tuple(stride)
    num_filters_in = inputs.get_shape()[-1]
    weights_shape = [kernel_h, kernel_w,
                     num_filters_out, num_filters_in]
    weights_initializer = tf.truncated_normal_initializer(stddev=stddev)
    l2_regularizer = None
    if weight_decay and weight_decay > 0:
      l2_regularizer = losses.l2_regularizer(weight_decay)
    weights = variables.variable('weights',
                                 shape=weights_shape,
                                 initializer=weights_initializer,
                                 regularizer=l2_regularizer,
                                 trainable=trainable,
                                 restore=restore)
    h = inputs.get_shape()[1].value
    w = inputs.get_shape()[2].value
    c = inputs.get_shape()[3].value
    output_shape = tf.concat(0, [tf.shape(inputs)[0:1], [h*stride[0], w*stride[1], num_filters_out]])
    conv = tf.nn.conv2d_transpose(inputs, weights, strides=[1, stride_h, stride_w, 1],
                                  output_shape=output_shape, padding=padding)
    conv.set_shape((None, h*stride[0], w*stride[1], num_filters_out))
    if batch_norm_params is not None:
      with scopes.arg_scope([batch_norm], is_training=is_training,
                            trainable=trainable, restore=restore):
        outputs = batch_norm(conv, **batch_norm_params)
    else:
      bias_shape = [num_filters_out,]
      bias_initializer = tf.constant_initializer(bias)
      biases = variables.variable('biases',
                                  shape=bias_shape,
                                  initializer=bias_initializer,
                                  trainable=trainable,
                                  restore=restore)
      outputs = tf.nn.bias_add(conv, biases)
    if activation:
      outputs = activation(outputs)
    return outputs

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
