# load up some dataset. Could be anything but skdata is convenient.
from skdata.cifar10.views import OfficialImageClassification
from tqdm import tqdm
import numpy as np
import tensorflow as tf

data = OfficialImageClassification()
tr_imgs = data.train.x
tr_labels = data.train.y
idx = range(len(tr_imgs))
np.random.shuffle(idx)
tr_imgs = tr_imgs[idx]
tr_labels = tr_labels[idx]

writer = tf.python_io.TFRecordWriter("cifar10_bird_train.tfrecords")
# iterate over each example
# wrap with tqdm for a progress bar
for example_idx in tqdm(range(len(tr_imgs))):
    img = tr_imgs[example_idx]
    label = tr_labels[example_idx]
    if label != 2:
        continue

    # construct the Example proto boject
    example = tf.train.Example(
        # Example contains a Features proto object
        features=tf.train.Features(
          # Features contains a map of string to Feature proto objects
          feature={
            # A Feature contains one of either a int64_list,
            # float_list, or bytes_list
            'label': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[label.astype("int64")])),
            'image': tf.train.Feature(
                int64_list=tf.train.Int64List(value=img.ravel().astype("int64"))),
    }))
    # use the proto object to serialize the example to a string
    serialized = example.SerializeToString()
    # write the serialized object to disk
    writer.write(serialized)
