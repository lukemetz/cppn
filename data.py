import tensorflow as tf
from tensorflow.python.ops import image_ops
import ipdb
import matplotlib.pylab as plt
from tqdm import trange, tqdm

def apply_transformations(image):
    return tf.cast(image, tf.float32) / 255.
    #image = tf.random_crop(image, [32, 32, 3])
    ##image = tf.image.per_image_whitening(image)

    #image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_flip_up_down(image)
    #image = tf.identity(image)
    #image = tf.image.random_flip_left_right(image)

    #image = tf.image.random_brightness(image, max_delta=0.2)
    #image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    #image = tf.image.random_hue(image, max_delta=0.02)
    #image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    return image


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    images_labels = []
    #for thread_idx in range(10):
    for thread_idx in range(8):
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'height': tf.FixedLenFeature([], tf.int64),
                'width':  tf.FixedLenFeature([], tf.int64),
                'depth': tf.FixedLenFeature([], tf.int64),
                'label': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string)
            })
        #label = tf.cast(features['label'], tf.float32)
        label = features['label']
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.reshape(image, (28, 28))
        #image = image_ops.decode_jpeg(features['image_raw'], channels=1)

        image = apply_transformations(image)
        images_labels.append([image, label])
    return images_labels

def get_inputs(batch_size, num_epochs=None, train=True):
    with tf.device("/cpu:0"):
        if train:
            filename = "train.tfrecords"
        else:
            filename = "validation.tfrecords"
        with tf.name_scope('data_input'):
            filename_queue = tf.train.string_input_producer([filename],
                                                            num_epochs=num_epochs)
            images_labels = read_and_decode(filename_queue)
            #return image, label
            images, sparse_labels = tf.train.shuffle_batch_join(
                images_labels, batch_size=batch_size,
                capacity=1000 + 3 * batch_size,
                min_after_dequeue=1000)
            return images, sparse_labels

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    with tf.Graph().as_default():

        images, labels = get_inputs(batch_size=128)
        i = tf.identity(images)
        i2 = tf.identity(images)
        init_op = tf.initialize_all_variables()
        #sess = tf.Session()
        sess = tf.Session(config=tf.ConfigProto(
            intra_op_parallelism_threads=2,
            inter_op_parallelism_threads=0,
            ))
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        #for x in trange(10000):
            #imgs, imgs2 = sess.run([i, i2])

        for _ in range(10):
            imgs, imgs2 = sess.run([i, i2])
            plt.subplot(3, 2, 1)
            plt.imshow(imgs[0])
            plt.subplot(3, 2, 2)
            plt.imshow(imgs[1])
            plt.subplot(3, 2, 3)
            plt.imshow(imgs[2])

            plt.subplot(3, 2, 4)
            plt.imshow(imgs2[0])
            plt.subplot(3, 2, 5)
            plt.imshow(imgs2[1])
            plt.subplot(3, 2, 6)
            plt.imshow(imgs2[2])
            plt.show()

        coord.request_stop()
        coord.join(threads)
        sess.close()
