"""This is the script to test the tf dataset function
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf
import time

tf.enable_eager_execution()


# Get original data
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data() 

#Others
# def load_and_preprocess_from_path_label(path, label):
#     return load_and_preprocess_image(path), label
# image_label_ds = ds.map(load_and_preprocess_from_path_label)
dataset = tf.data.Dataset.from_tensor_slices( (tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32), 
tf.cast(mnist_labels,tf.int64))) 
dataset = dataset.shuffle(1000).batch(32) 

image_batch, label_batch = next(iter(dataset))
print('Image batch shape:', image_batch.shape)
print('Label batch shape:', label_batch.shape)
