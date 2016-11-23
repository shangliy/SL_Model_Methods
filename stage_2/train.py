#author: Shanglin Yang(kudoysl@gmail.com)
"""
To generate the triplet input for the slim3 structure
"""
import os
import sys
import datetime
import numpy as np
import tensorflow as tf
import inception.inception_v3 as inception
from scipy.misc import imsave
from inception.slim import slim

from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util



import data_util

H = 299
W = 299
DATA_INPUT = "triplet_fine_data.json"
CLASS_NUM = 6
BATCH_SIZE = 1
MARGIN = 0.1
MAX_INTERATION = 1

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '../model/model.ckpt-157585',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")

#Feature includes three images as input
anchor_Input = tf.placeholder(tf.float32, shape=[H,W,3])
positi_Input = tf.placeholder(tf.float32, shape=[H,W,3])
negati_Input = tf.placeholder(tf.float32, shape=[H,W,3])

anchor_Label = tf.placeholder(tf.int32, shape=[1])
positi_Label = tf.placeholder(tf.int32, shape=[1])
negati_Label = tf.placeholder(tf.int32, shape=[1])

dtypes = [tf.float32, tf.float32, tf.float32, tf.int32,tf.int32,tf.int32]
shapes = (
            [H, W, 3],
            [H, W, 3],
            [H, W, 3],
            [1],
            [1],
            [1]
        )

q = tf.FIFOQueue(capacity=30, dtypes=dtypes, shapes=shapes)
enqueue_op = q.enqueue([anchor_Input,positi_Input,negati_Input,anchor_Label,positi_Label,negati_Label])

def make_feed(d):

  return {anchor_Input: d['train_anchor'], positi_Input: d['train_positive'], negati_Input: d['train_negative'],\
            anchor_Label: d['label_anchor'], positi_Label: d['label_positive'], negati_Label: d['label_negative']}

def thread_loop(sess, enqueue_op, gen):
    for d in gen:
        sess.run(enqueue_op, feed_dict=make_feed(d))
def compute_euclidean2_distance(x,y):
    """
    Computer the euclidean square distance between two tensors

    """
    d = tf.square(tf.sub(x,y))
    return d

def compute_cos_distance(x,y):
    """
    Computer the cos distance between two tensors
    """
    d = tf.reduce_sum(tf.mul(x,y))
    return d

def compute_loss(anchor_feature,positive_feature,negative_feature,\
                    anchor_logit,positive_logit,negative_logit,\
                    anchor_label,positive_label,negative_label,margin):
    """
    Compute the triplet loss
    """
    with tf.name_scope("triplet_loss"):
        #d_p = compute_euclidean2_distance(anchor_feature,positive_feature)
        #d_n = compute_euclidean2_distance(anchor_feature,negative_feature)
        d_p = compute_cos_distance(anchor_feature,positive_feature)
        d_n = compute_cos_distance(anchor_feature,negative_feature)

        loss_tri = tf.reduce_mean(tf.maximum(0., (d_p-d_n)+margin)) #Triplet loss (not good) need another to fix it
         # Define a non-class loss (could we us)


    with tf.name_scope("classification_loss"):
        labels_anchor = tf.to_int64(anchor_label)
        labels_positive = tf.to_int64(positive_label)
        labels_negative = tf.to_int64(negative_label)
        anchor_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            anchor_logit, labels_anchor, name='anchor_xentropy')
        positive_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            positive_logit, labels_positive, name='positive_xentropy')
        negative_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            negative_logit, labels_negative, name='negative_xentropy')
        anchor_loss = tf.reduce_mean(anchor_cross_entropy, name='anchor_xentropy_mean')
        positive_loss = tf.reduce_mean(positive_cross_entropy, name='positive_xentropy_mean')
        negative_loss = tf.reduce_mean(negative_cross_entropy, name='negative_xentropy_mean')

        loss_class = anchor_loss + positive_loss + negative_loss

    loss = loss_tri + loss_class

    return loss, tf.reduce_mean(d_p),tf.reduce_mean(d_n)

anchor_tensor,positive_tensor,negative_tensor,anchor_label,positive_label,negative_label = q.dequeue_many(BATCH_SIZE)

_,logit_anchor,feature_anchor = inception.inference(anchor_tensor, CLASS_NUM, for_training=True,restore_logits=False,scope="Triplet")
_,logit_positive,feature_positive = inception.inference(positive_tensor, CLASS_NUM, for_training=True,restore_logits=False,scope="Triplet",reuse=True)
_,logit_nagative,feature_negative = inception.inference(negative_tensor, CLASS_NUM, for_training=True,restore_logits=False,scope="Triplet",reuse=True)


#print (feature_anchor.get_shape())
anchor_label = tf.reshape(anchor_label,[BATCH_SIZE,])
positive_label = tf.reshape(positive_label,[BATCH_SIZE,])
negative_label = tf.reshape(negative_label,[BATCH_SIZE,])
print(anchor_label.get_shape())
loss, positives, negatives = compute_loss(feature_anchor, feature_positive, feature_negative,\
                                            logit_anchor, logit_positive, logit_nagative,\
                                            anchor_label,positive_label,negative_label,\
                                            MARGIN)


    # Defining training parameters
batch = tf.Variable(0)
learning_rate = tf.train.exponential_decay(
        0.001, # Learning rate
        batch * BATCH_SIZE,
        MAX_INTERATION * BATCH_SIZE * 3,
        0.95 # Decay step
    )

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch)



with tf.Session() as sess:

        train_writer = tf.train.SummaryWriter('./logs_tensorboard/triplet/train',sess.graph)
        test_writer = tf.train.SummaryWriter('./logs_tensorboard/triplet/test',sess.graph)

        tf.scalar_summary('loss', loss)
        tf.scalar_summary('positives', positives)
        tf.scalar_summary('negatives', negatives)
        tf.scalar_summary('lr', learning_rate)
        merged = tf.merge_all_summaries()

        gen = data_util.load_data(DATA_INPUT,CLASS_NUM,W,H)
        #run once in case
        d = gen.next()
        sess.run(enqueue_op, feed_dict=make_feed(d))
        t = tf.train.threading.Thread(target=thread_loop,
                                            args=(sess, enqueue_op,gen))
        t.daemon = True
        t.start()

        tf.initialize_all_variables().run()

        if FLAGS.pretrained_model_checkpoint_path:
            assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
        variables_to_restore = tf.get_collection(slim.variables.VARIABLES_TO_RESTORE)
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)
        print('Pre-trained model restored from %s' %
                ( FLAGS.pretrained_model_checkpoint_path))

        for step in range(MAX_INTERATION):
            train_data,_, l, lr, summary = sess.run([anchor_tensor,optimizer, loss, learning_rate, merged])
            train_writer.add_summary(summary, step)
            #imsave(str(step)+".jpg",train_data[0])
            print("Loss Validation {0}".format(l))
            #print(train_image[0][0])
        output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['Triplet/logits/flatten/Reshape'])
        with gfile.FastGFile('./triplet_1.pb', 'wb') as f:
            f.write(output_graph_def.SerializeToString())
