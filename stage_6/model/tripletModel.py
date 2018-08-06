"""
Model
"""
import sys
import tensorflow as tf
sys.path.append('models/research/slim')
from nets.mobilenet import mobilenet_v2

class tripletModel():
    def __init__(self, config):
        self.embidding_dim = config.embedding_dim
        self.margin = config.margin
        self.build_model()
        self.init_saver()


    def build_model(self, input, is_training=True):
        
        with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=is_training)):
            logits, endpoints = mobilenet_v2.mobilenet(images)
        
        self.mobilenetfea = endpoints["global_pool"]

        # network architecture
        with tf.name_scope("embedding"):
            self.visualembedding = tf.layers.dense(self.mobilenetfea, self.embidding_dim, 
                                                regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                                activation=tf.nn.relu, name="visualembedding")
    
        with tf.name_scope("loss"):
            self.tri_loss = self.triplet_loss()
            tf.summary.scalar('triplet loss', self.tri_loss)
            self.reg_loss = tf.losses.get_regularization_losses()
            self.total_loss = self.tri_loss + self.reg_loss           
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.total_loss,
                                                                                         global_step=self.global_step_tensor)

    
    def triplet_loss(self):

        embedding_anchor, embedding_positive, embedding_negative 
                            = tf.split(self.visualembedding, num_or_size_splits=3)
        normalize_a = tf.nn.l2_normalize(embedding_anchor,3)        
        normalize_p = tf.nn.l2_normalize(embedding_positive,3)
        normalize_n = tf.nn.l2_normalize(embedding_anchor,3)        
        
        cos_similarity_a_p =tf.reduce_sum(tf.multiply(normalize_a,normalize_p))
        cos_similarity_a_n =tf.reduce_sum(tf.multiply(normalize_a,normalize_n))

        triplet_loss = tf.maximum(cos_similarity_a_n - cos_similarity_a_p + self.margin, 0.0)

        return triplet_loss

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)