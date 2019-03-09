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
BATCH_SIZE = 30  # 3 * N
MARGIN = 0.1
MAX_INTERATION = 100

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '../model/model.ckpt-157585',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")

#Feature includes three images as input
image_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE,H,W,3])
image_label = tf.placeholder(tf.int32, shape=[BATCH_SIZE,1])

dtypes = [tf.float32,tf.int32]
shapes = (
            [BATCH_SIZE,H, W, 3],
            [BATCH_SIZE,1]
        )

q = tf.FIFOQueue(capacity=30, dtypes=dtypes, shapes=shapes)
enqueue_op = q.enqueue([image_data ,image_label])

def make_feed(d):

  return {image_data: d['image_data'],  image_label: d['image_label']}

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
    normed_x = tf.nn.l2_normalize(x, dim=1)
    normed_y = tf.nn.l2_normalize(y, dim=1)

    sim = tf.matmul(normed_x, tf.transpose(y, [1, 0]))
    d = tf.reduce_sum(sim)
    return d

def compute_loss(input_feature,\
                    input_logit,\
                    input_label,margin):
    """
    Compute the triplet loss
    """
    triplet_num = BATCH_SIZE/3
    ori_index = np.zeros(triplet_num)
    pos_index = np.zeros(triplet_num)
    neg_index = np.zeros(triplet_num)
    for i in range(triplet_num):
        ori_index[i] = 3*i
        pos_index[i] = (3*i+1)
        neg_index[i] = (3*i+2)

    ori_index_ten = tf.constant(ori_index,dtype="int32")
    pos_index_ten = tf.constant(pos_index,dtype="int32")
    neg_index_ten = tf.constant(neg_index,dtype="int32")

    with tf.name_scope("triplet_loss"):

        anchor_feature = tf.gather(input_feature,ori_index_ten)
        positive_feature = tf.gather(input_feature,pos_index_ten)
        negative_feature = tf.gather(input_feature,neg_index_ten)

        #d_p = compute_euclidean2_distance(anchor_feature,positive_feature)
        #d_n = compute_euclidean2_distance(anchor_feature,negative_feature)
        s_p = compute_cos_distance(anchor_feature,positive_feature)
        s_n = compute_cos_distance(anchor_feature,negative_feature)

        loss_tri = tf.reduce_mean(tf.maximum(0., (s_n-s_p)+margin)) #Triplet loss (not good) need another to fix it
         # Define a non-class loss (could we us)

    loss = loss_tri

    return loss, tf.reduce_mean(s_p),tf.reduce_mean(s_n)

anchor_tensor,anchor_label = q.dequeue()
anchor_tensor = tf.reshape(anchor_tensor,[BATCH_SIZE,H, W, 3])
anchor_label = tf.reshape(anchor_label,[BATCH_SIZE,1])
print(anchor_tensor.get_shape())

_,logit_anchor,feature_anchor = inception.inference(anchor_tensor, CLASS_NUM, for_training=True,restore_logits=False,scope="Triplet")


#print (feature_anchor.get_shape())
anchor_label = tf.reshape(anchor_label,[BATCH_SIZE,])

loss, positives, negatives = compute_loss(feature_anchor, logit_anchor, anchor_label, MARGIN)


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

        gen = data_util.load_data(DATA_INPUT,CLASS_NUM,W,H,BATCH_SIZE)
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
        with gfile.FastGFile('./triplet_1000.pb', 'wb') as f:
            f.write(output_graph_def.SerializeToString())
