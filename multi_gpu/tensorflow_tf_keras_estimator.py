rom __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app as absl_app
import tensorflow as tf
from tensorflow import Graph
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.framework.python.ops import arg_scope

from tensorflow.keras import layers
from official.utils.misc import distribution_utils

import dataset as dataset
slim = tf.contrib.slim

BATCH_NORM_DECAY = 0.996
BATCH_NORM_EPSILON = 1e-3


LEARNING_RATE = 1e-4
NUM_GPUS = 1
MODEL_DIR = './logs/'
DATA_DIR = './data/'
BUFFER_SIZE = 1000
BATCH_SIZE = 16
EPOCH_BETWEEN_EVALS = 1
TRAIN_EPOCHS = 2
CLASS_NUM = 9
TRAIN_DIR= \
        '/home/shangliy/Projects/KAMI/kami/Img2Txt/dataacquision/jd_data_fromteresa/NoisyPattern_Data/'
EVAL_DIR = \
        '/home/shangliy/Projects/KAMI/kami/Img2Txt/dataacquision/jd_data_fromteresa/pattern_val_images/'

from resnet_v1 import resnet_v1
#with tf.Graph().as_default():
#    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False) 

def create_model():
    
    #resenet_feature = base_model.output
    #feature = tf.keras.Model.GlobalAveragePooling2D(name='global_avg_pool')(resenet_feature)
    #prediction = tf.keras.layers.Dense(CLASS_NUM,
    #                                   name='predictions')(feature)
    #model = tf.keras.Model(inputs=base_model.input, outputs=prediction)

    model = resnet_v1((224,224,3), 20, num_classes=9)
    return model


def model_fn(features, labels, mode, params):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    is_eval = (mode == tf.estimator.ModeKeys.EVAL)
    global_model = create_model()
    image = features
    if isinstance(image, dict):
        image = features['image']

    logits = global_model(image, training=is_training)
        
    if mode == tf.estimator.ModeKeys.PREDICT:
                
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits),
        }
        return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.PREDICT,
                predictions=predictions,
                export_outputs={
                        'classify': tf.estimator.export.PredictOutput(predictions)
            })
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        accuracy = tf.metrics.accuracy(
                labels=labels, predictions=tf.argmax(logits, axis=1))

        # Name tensors to be logged with LoggingTensorHook.
        tf.identity(LEARNING_RATE, 'learning_rate')
        tf.identity(loss, 'cross_entropy')
        tf.identity(accuracy[1], name='train_accuracy')

        # Save accuracy scalar to Tensorboard output.
        tf.summary.scalar('train_accuracy', accuracy[1])

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))
                
    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.EVAL,
                loss=loss,
                eval_metric_ops={
                    'accuracy':
                        tf.metrics.accuracy(
                            labels=labels, predictions=tf.argmax(logits, axis=1)),
                })

def run_():
    with tf.Graph().as_default():
        model_function = model_fn

        session_config = tf.ConfigProto(
        #inter_op_parallelism_threads=flags_obj.inter_op_parallelism_threads,
        #intra_op_parallelism_threads=flags_obj.intra_op_parallelism_threads,
        allow_soft_placement=True)

        distribution_strategy = distribution_utils.get_distribution_strategy(
            distribution_strategy='default',
            num_gpus=NUM_GPUS,
            all_reduce_alg='True')
        
        run_config = tf.estimator.RunConfig(
        train_distribute=distribution_strategy, session_config=session_config)
            
        _classifier = tf.estimator.Estimator(
            model_fn=model_function,
            model_dir=MODEL_DIR,
            config=run_config)
            
            

        # Set up training and evaluation input functions.
        def train_input_fn():
            """Prepare data for training."""

            # When choosing shuffle buffer sizes, larger sizes result in better
            # randomness, while smaller sizes use less memory. MNIST is a small
            # enough dataset that we can easily shuffle the full epoch.
            ds = dataset.train(TRAIN_DIR)
            ds = ds.cache().shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE)

            # Iterate through the dataset a set number (`epochs_between_evals`) of times
            # during each training session.
            ds = ds.repeat(EPOCH_BETWEEN_EVALS)
            return ds
            
        def eval_input_fn():
            return dataset.test(EVAL_DIR).batch(
            BATCH_SIZE).make_one_shot_iterator().get_next()
            
        # Train and evaluate model.
        for _ in range(TRAIN_EPOCHS // EPOCH_BETWEEN_EVALS):
            _classifier.train(input_fn=train_input_fn)
            #tf.keras.backend.clear_session()
            eval_results = _classifier.evaluate(input_fn=eval_input_fn)
            print('\nEvaluation results:\n\t%s\n' % eval_results)

def main(_):
    run_()

if __name__ == '__main__':
    absl_app.run(main)