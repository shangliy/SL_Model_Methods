import tensorflow as tf
import numpy as np
import pickle
import math
import time
import os
import sys
from sklearn.model_selection import train_test_split

TRAIN_DATA_PATH = './data/train.p'
TEST_DATA_PATH = './data/test.p'

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'

batch_size = 100
MAXIMUM_INPUT = 100000
IMAGE_PIXELS = 32*32*3
NUM_CLASSES = 43
max_steps = 100000
learning_rate = 0.01


def model(images, hidden1_units, hidden2_units,NUM_CLASSES):
  """Build the MNIST model up to where it may be used for inference.

  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  # Hidden 1
  with tf.name_scope('hidden1'):
    layer1_weights = tf.Variable(
        tf.truncated_normal([3, 3, 3, hidden1_units],
                            stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
        name='weights')
    layer1_biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
    hidden1 = tf.nn.relu(tf.nn.conv2d(images, layer1_weights, [1, 1, 1, 1], padding='SAME')+layer1_biases)
  hidden1_pool = tf.nn.max_pool(hidden1, ksize=[1, 2, 2, 1],strides=[1, 1,1, 1], padding='SAME',name='hidden1_pool')
  # Hidden 2
  with tf.name_scope('hidden2'):
    layer2_weights = tf.Variable(
        tf.truncated_normal([3, 3, hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights')
    layer2_biases = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases')
    hidden2 = tf.nn.relu(tf.nn.conv2d(hidden1_pool, layer2_weights, [1, 2, 2, 1], padding='SAME')+layer2_biases)

  hidden2_pool = tf.nn.max_pool(hidden2, ksize=[1, 2, 2, 1],strides=[1, 2,2, 1], padding='SAME',name='hidden2_pool')
  # fully_connected_
  with tf.name_scope('fully_connected'):
    input_shape = hidden2_pool.get_shape().as_list()
    print(input_shape)
    dim = input_shape[1]*input_shape[2]*input_shape[3]
    inputs_transposed = tf.transpose(hidden2_pool,(0,3,1,2))
    hidden2_flat = tf.reshape(inputs_transposed,[-1,dim])
    weights_fc = tf.Variable(
         tf.truncated_normal([dim, 100],
                              stddev=1.0 / math.sqrt(float(hidden2_units))),
          name='weights')
    biases_fc = tf.Variable(tf.zeros([100]),
                           name='biases')
    fc = tf.matmul(hidden2_flat, weights_fc) + biases_fc

  # Linear
  with tf.name_scope('softmax_linear'):
    fc_shape = fc.get_shape()
    weights = tf.Variable(
        tf.truncated_normal([100, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    logits = tf.matmul(fc, weights) + biases
  return logits

def loss_fun(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss

def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))

def load_data():
    with open(TRAIN_DATA_PATH, mode='rb') as f:
        train_data = pickle.load(f)
        X_train, y_train = train_data['features'], train_data['labels']
    with open(TEST_DATA_PATH, mode='rb') as f:
        test_data = pickle.load(f)
        X_test, y_test = test_data['features'], test_data['labels']

    IMAGE_SIZE = X_train.shape[1]
    data_size = len(X_train)
    print("IAMGE_shape is:" + str(IMAGE_SIZE))
    print("data_size is:" + str(data_size))

    return X_train, y_train,X_test, y_test

def prepro(X_train,X_test):
    X_train = X_train*2/255-1
    X_test = X_test*2/255-1
    #X_train = X_train.reshape(X_train.shape[0],IMAGE_PIXELS)
    #X_test = X_test.reshape(X_test.shape[0],IMAGE_PIXELS)
    return X_train,X_test

def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            batch_data,batch_labels):
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  #steps_per_epoch = data_set.num_examples  // FLAGS.batch_size
  num_examples = batch_data.shape[0]

  feed_dict = {images_placeholder : batch_data, labels_placeholder : batch_labels}
  true_count = sess.run(eval_correct, feed_dict=feed_dict)
  precision = true_count / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))

def main():
    X_train, y_train,X_test, y_test = load_data()
    X_train, X_test = prepro(X_train,X_test)
    print (len(set(y_train)))
    NUM_CLASSES = (len(set(y_test)))
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=0)
    print (len(set(y_train)))
    if (len(set(y_train)) != NUM_CLASSES):
        sys.exit()


    with tf.Graph().as_default():

        images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,32,32,3))
        labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
        logits = model(images_placeholder,128,32,NUM_CLASSES)
        loss = loss_fun(logits, labels_placeholder)
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        eval_correct = evaluation(logits, labels_placeholder)
        #summary = tf.summary

        init = tf.initialize_all_variables()


        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.train.SummaryWriter('./work/', sess.graph)

        sess.run(init)
        for step in range(max_steps):
              start_time = time.time()
              '''
              offset = (step * batch_size) % (X_train.shape[0] - batch_size)
              batch_data_train = X_train[offset:(offset + batch_size), :]
              batch_labels_train = y_train[offset:(offset + batch_size)]
              batch_data_val = X_valid[offset:(offset + batch_size), :]
              batch_labels_val = y_valid[offset:(offset + batch_size)]
              batch_data_test = X_test[offset:(offset + batch_size), :]
              batch_labels_test = y_test[offset:(offset + batch_size)]
              '''
              train_index = np.random.randint(X_train.shape[0], size=batch_size)


              batch_data_train = X_train[train_index, :,:,:]
              batch_labels_train = y_train[train_index]



              feed_dict = {images_placeholder : batch_data_train, labels_placeholder : batch_labels_train}
              _, loss_value = sess.run([train_op, loss],
                                       feed_dict=feed_dict)
              duration = time.time() - start_time
              # Write the summaries and print an overview fairly often.
              if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                #summary_str = sess.run(summary, feed_dict=feed_dict)
                #summary_writer.add_summary(summary_str, step)
                #summary_writer.flush()

              # Save a checkpoint and evaluate the model periodically.
              if (step + 1) % 1000 == 0 or (step + 1) == max_steps:
                checkpoint_file = os.path.join('./work', 'model.ckpt')
                #saver.save(sess, checkpoint_file, global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        batch_data_train,batch_labels_train)
                # Evaluate against the validation set.
                valid_index = np.random.randint(X_valid.shape[0], size=100)
                batch_data_val = X_valid[valid_index , :,:,:]
                batch_labels_val = y_valid[valid_index]

                print('Validation Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        batch_data_val,batch_labels_val)
                # Evaluate against the test set.

                print('Test Data Eval:')
                test_index = np.random.randint(X_test.shape[0], size=100)
                batch_data_test = X_test[test_index, :,:,:]
                batch_labels_test = y_test[test_index]
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        batch_data_test,batch_labels_test)






if __name__ == "__main__":
    main()
