#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST

from util import *

class model(object):

    def __init__(self,
                 conv1_kernel_size=3,
                 conv1_output=108,

                 conv2_kernel_size=3,
                 conv2_output=200,

                 fc1_output=600,
                 n_classes=43,

                 seed=10, image_size =[32,32,3], use_gpu = False):
        """
        Create all the necessary variables for this CNN
        **Parameters**
            conv1_kernel_size=5,
            conv1_output=32,
            conv2_kernel_size=5,
            conv2_output=64,
            fc1_output=400,
            n_classes=43
            seed = 10
        """
        # First convolutional
        self.W_conv1 = create_weight_variables([conv1_kernel_size, conv1_kernel_size, image_size[2], conv1_output], seed=seed, name="W_conv1", use_gpu=use_gpu)
        self.b_conv1 = create_bias_variables([conv1_output], name="bias_conv1", use_gpu=use_gpu)

        # Second convolutional
        self.W_conv2 = create_weight_variables([conv2_kernel_size, conv2_kernel_size, conv1_output, conv2_output], seed=seed, name="W_conv2", use_gpu=use_gpu)
        self.b_conv2 = create_bias_variables([conv2_output], name="bias_conv2", use_gpu=use_gpu)

        # First fc
        self.W_fc1 = create_weight_variables([(image_size[0] // 4) * (image_size[1] // 4) * conv2_output+((image_size[0] // 2) * (image_size[1] // 2) * conv1_output), fc1_output], seed=seed, name="W_fc1", use_gpu=use_gpu)
        self.b_fc1 = create_bias_variables([fc1_output], name="bias_fc1", use_gpu=use_gpu)

        # Second FC fc
        self.W_fc2 = create_weight_variables([fc1_output, n_classes], seed=seed, name="W_fc2", use_gpu=use_gpu)
        self.b_fc2 = create_bias_variables([n_classes], name="bias_fc2", use_gpu=use_gpu)

        self.seed = seed

    def create_network(self, data, train=True):
        """
        Create the Lenet Architecture
        **Parameters**
          data: Input data
          train:
        **Returns
          features_back: Features for backpropagation
          features_val: Features for validation
        """

        # Creating the architecture
        # First convolutional
        with tf.name_scope('conv_1') as scope:
            conv1 = create_conv2d(data, self.W_conv1)
        # relu1 = create_relu(conv1, self.b_conv1)
        # relu1 = create_sigmoid(conv1, self.b_conv1)

        with tf.name_scope('tanh_1') as scope:
            tanh_1 = create_tanh(conv1, self.b_conv1)


        # Pooling
        # pool1 = create_max_pool(relu1)
        # pool1 = create_max_pool(relu1)
        with tf.name_scope('pool_1') as scope:
            pool1 = create_max_pool(tanh_1)


        # Second convolutional
        with tf.name_scope('conv_2') as scope:
            conv2 = create_conv2d(pool1, self.W_conv2)
        # relu2 = create_relu(conv2, self.b_conv2)
        # relu2 = create_sigmoid(conv2, self.b_conv2)


        with tf.name_scope('tanh_2') as scope:
            # pool2 = create_max_pool(relu2)
            #tanh_2 = create_relu(conv2, self.b_conv2)
            # pool2 = create_max_pool(conv2)
            tanh_2 = create_tanh(conv2, self.b_conv2)

        # Pooling
        with tf.name_scope('pool_2') as scope:
            pool2 = create_max_pool(tanh_2)


        #if train:
            #pool2 = tf.nn.dropout(pool2, 0.5, seed=self.seed)

        # Reshaping all the convolved images to 2D to feed the FC layers
        # FC1
        with tf.name_scope('fc_1') as scope:
            pool1_shape = pool1.get_shape().as_list()
            reshape1 = tf.reshape(pool1, [pool1_shape[0], pool1_shape[1] * pool1_shape[2] * pool1_shape[3]])
            pool2_shape = pool2.get_shape().as_list()
            reshape2 = tf.reshape(pool2, [pool2_shape[0], pool2_shape[1] * pool2_shape[2] * pool2_shape[3]])
            reshape = tf.concat(1,[reshape1,reshape2])
            #fc1 = tf.nn.relu(tf.matmul(reshape, self.W_fc1) + self.b_fc1)
            fc1 = tf.nn.tanh(tf.matmul(reshape, self.W_fc1) + self.b_fc1)

        #if train:
            #fc1 = tf.nn.dropout(fc1, 0.5, seed=self.seed)

        # FC2
        with tf.name_scope('fc_2') as scope:
            fc2 = tf.matmul(fc1, self.W_fc2) + self.b_fc2
            #fc2 = tf.nn.softmax(tf.matmul(fc1, self.W_fc2) + self.b_fc2)

        return fc2
