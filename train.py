#!/usr/bin/env python
# @author ShanglinYang (kudoysl@gmail.com)
import sys
import tensorflow as tf
from docopt import docopt

import util
import model
import DataStr

TRAIN_DATA_PATH = './data/train.p'
TEST_DATA_PATH = './data/test.p'
IMAGE_SIZE = [32, 32, 3]
CLASS_NUMS = 43
TRIPLET_TRAINING = False

def compute_euclidean2_distance(x,y):
    """
    Computer the euclidean square distance between two tensors

    """

    d = tf.square(tf.sub(x,y))

    return d 

def triplet_loss(anchor_feature,positive_feature,negative_feature,margin):
    """
    Compute the triplet loss
    """
    with tf.name_scope("triplet_loss"):
        d_p = compute_euclidean2_distance(anchor_feature,positive_feature)
        d_n = compute_euclidean2_distance(anchor_feature,negative_feature)

        loss = tf.reduce_mean(tf.maximum(0., d_p-d_n+margin))


        return loss, tf.reduce_mean(d_p),tf.reduce_mean(d_n)

def classification_loss(logits, labels):
    """
    Compute the classification loss
    """
    with tf.name_scope("classification_loss"):
        labels = tf.to_int64(labels)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, labels, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

    return loss

def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))

def do_eval(sess,
            eval_correct,
            num_examples,
            feed_dict):
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  #steps_per_epoch = data_set.num_examples  // FLAGS.batch_size
  true_count = sess.run(eval_correct, feed_dict=feed_dict)
  precision = true_count / num_examples
  print('  Num examples: %d  Num correct: %d  Precision: %0.04f' %
        (num_examples, true_count, precision))


def main():
    '''
    args = docopt(__doc__, version='Mnist training with TensorFlow')
    
    BATCH_SIZE = int(args['--barch-size'])
    ITERATIONS = int(args['--iterations'])
    VALIDATION_TEST = int(args['--validation-test'])
    '''
    BATCH_SIZE = 1000
    ITERATIONS = 5000
    VALIDATION_TEST = 100
    MARGIN = 0.01
    SEED = 10 

    train_data,train_label,test_data,test_label = util.load_data(TRAIN_DATA_PATH,TEST_DATA_PATH) 
    print(train_data[0][0][0][0])
    sys.exit()
    data_shuffler_train = DataStr.DataStr(train_data, train_label, IMAGE_SIZE)
    data_shuffler_test = DataStr.DataStr(test_data, test_label, IMAGE_SIZE)

    train_anchor_data = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]), name="anchor")
    labels_anchor = tf.placeholder(tf.int32, shape=BATCH_SIZE)

    if TRIPLET_TRAINING:
        train_positive_data = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]), name="positive")
        train_negative_data = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]), name="negative")
        
        labels_positive = tf.placeholder(tf.int32, shape=BATCH_SIZE)
        labels_negative = tf.placeholder(tf.int32, shape=BATCH_SIZE)

    cnn_model = model.model(seed=SEED,image_size=IMAGE_SIZE,n_classes=CLASS_NUMS)
    train_anchor = cnn_model.create_network(train_anchor_data,)
    if TRIPLET_TRAINING:
        train_positive = cnn_model.create_network(train_positive_data)
        train_negative = cnn_model.create_network(train_negative_data)

    if TRIPLET_TRAINING:
        loss, positives, negatives = triplet_loss(train_anchor, train_positive, train_negative, MARGIN)
    else:
        loss =  classification_loss(train_anchor, labels_anchor)

    # Defining training parameters
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.01, # Learning rate
        batch * BATCH_SIZE,
        data_shuffler_train.train_data.shape[0],
        0.95 # Decay step
    )

    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=batch)

    eval_correct = evaluation(train_anchor, labels_anchor)

    # Training
    with tf.Session() as session:
        train_writer = tf.train.SummaryWriter('./logs_tensorboard/triplet/train',session.graph)

        test_writer = tf.train.SummaryWriter('./logs_tensorboard/triplet/test',session.graph)

        tf.scalar_summary('lr', learning_rate)
        tf.scalar_summary('loss', loss)

        if TRIPLET_TRAINING:
            tf.scalar_summary('positives', positives)
            tf.scalar_summary('negatives', negatives)
        
        merged = tf.merge_all_summaries()


        tf.initialize_all_variables().run()
        
        for step in range(ITERATIONS):

            batch_anchor, batch_positive, batch_negative, \
            batch_labels_anchor, batch_labels_positive, batch_labels_negative = \
                data_shuffler_train.get_triplet(n_labels=CLASS_NUMS, n_triplets=BATCH_SIZE)
            if TRIPLET_TRAINING:
                feed_dict = {train_anchor_data: batch_anchor,
                         train_positive_data: batch_positive,
                         train_negative_data: batch_negative,
                }
            else:
                feed_dict = {
                            train_anchor_data: batch_anchor,
                            labels_anchor: batch_labels_anchor,
                         }


            _, l, lr, summary = session.run([optimizer, loss, learning_rate, merged],
                                                feed_dict=feed_dict)
            train_writer.add_summary(summary, step)

            if step % VALIDATION_TEST == 0:

                batch_anchor, batch_positive, batch_negative, \
                batch_labels_anchor, batch_labels_positive, batch_labels_negative = \
                    data_shuffler_train.get_triplet(n_labels=CLASS_NUMS, n_triplets=BATCH_SIZE, is_target_set_train=False)

                if TRIPLET_TRAINING:
                    feed_dict = {train_anchor_data: batch_anchor,
                            train_positive_data: batch_positive,
                            train_negative_data: batch_negative,
                    }
                else:
                    feed_dict = {
                                train_anchor_data: batch_anchor,
                                labels_anchor: batch_labels_anchor,
                    }
                lv, summary = session.run([loss, merged], feed_dict=feed_dict)
                test_writer.add_summary(summary, step)
                print('Validation Data Eval:')
                print("Step {0}:Loss Validation {1}".format(step,lv))
                
                do_eval(session,eval_correct,BATCH_SIZE,feed_dict)

                batch_anchor, batch_positive, batch_negative, \
                batch_labels_anchor, batch_labels_positive, batch_labels_negative = \
                    data_shuffler_test.get_triplet(n_labels=CLASS_NUMS, n_triplets=BATCH_SIZE)

                if TRIPLET_TRAINING:
                    feed_dict = {train_anchor_data: batch_anchor,
                            train_positive_data: batch_positive,
                            train_negative_data: batch_negative,
                    }
                else:
                    feed_dict = {
                                train_anchor_data: batch_anchor,
                                labels_anchor: batch_labels_anchor,
                    }
                lv, summary = session.run([loss, merged], feed_dict=feed_dict)
                test_writer.add_summary(summary, step)
                print('Test Data Eval:')
                print("Step {0}:Loss Validation {1}".format(step,lv))              
                do_eval(session,eval_correct,BATCH_SIZE,feed_dict)



if __name__ == "__main__":
    main()
