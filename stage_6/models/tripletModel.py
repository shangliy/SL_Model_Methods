"""
Model
"""
import sys
import tensorflow as tf
from models.mobilenet import mobilenet_v2

def _pairwise_distances(embeddings_1, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances

class tripletModel():
    def __init__(self, config):
        self.config = config
        self.embidding_dim = config.embedding_dim
        self.margin = config.margin
        self.input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        # global tensors
        self.cur_epoch_tensor = None
        self.cur_epoch_input = None
        self.cur_epoch_assign_op = None

        self.global_step_tensor = None
        self.global_step_input = None
        self.global_step_assign_op = None

        self.init_global_step()
        self.init_cur_epoch()

        self.build_model()
        self.init_saver()

        # init the global step, global time step, the current epoch and the summaries

    def build_model(self, is_training=True):

        with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=self.is_training)):
            _, endpoints = mobilenet_v2.mobilenet(self.input)

        self.mobilenetfea = endpoints["global_pool"]

        # network architecture
        with tf.variable_scope("embedding"):
            #mobilefea_squeeze= tf.squeeze(self.mobilenetfea )
            #print(mobilefea_squeeze)
            self.visualembedding = tf.layers.dense(self.mobilenetfea, self.embidding_dim,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                                activation=tf.nn.relu, name="visualembedding")

        with tf.variable_scope("loss"):
            self.tri_loss = self.triplet_loss()
            tf.summary.scalar('triplet loss', self.tri_loss)
            self.reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses())
            tf.summary.scalar('triplet loss', self.reg_loss)
            self.total_loss = self.tri_loss + self.reg_loss
            self.train_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.total_loss,
                                                                                         global_step=self.global_step_tensor)
            self.merged = tf.summary.merge_all()



    def triplet_loss(self):

        embedding_anchor, embedding_positive, embedding_negative = tf.split(self.visualembedding, num_or_size_splits=3)

        em_anchor =tf.squeeze(embedding_anchor)
        em_positi =tf.squeeze(embedding_positive)
        em_negati =tf.squeeze(embedding_negative)

        #normalize_a = tf.nn.l2_normalize(embedding_anchor,3)
        #normalize_p = tf.nn.l2_normalize(embedding_positive,3)
        #normalize_n = tf.nn.l2_normalize(embedding_negative,3)

        #cos_similarity_a_p = tf.reduce_sum(tf.multiply(normalize_a,normalize_p))
        #cos_similarity_a_n = tf.reduce_sum(tf.multiply(normalize_a,normalize_n))

        self.euclidean_a_p = tf.reduce_sum(tf.square(em_anchor-em_positi),1)
        euclidean_a_n = tf.reduce_sum(tf.square(em_anchor-em_negati),1)

        triplet_loss = tf.maximum(self.euclidean_a_p - euclidean_a_n + self.margin, 0.0)
        triplet_loss = tf.reduce_mean(triplet_loss)

        return triplet_loss

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def init_cur_epoch(self):
        """
        Create cur epoch tensor to totally save the process of the training
        :return:
        """
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.cur_epoch_input = tf.placeholder('int32', None, name='cur_epoch_input')
            self.cur_epoch_assign_op = self.cur_epoch_tensor.assign(self.cur_epoch_input)

    def init_global_step(self):
        """
        Create a global step variable to be a reference to the number of iterations
        :return:
        """
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
            self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)


    def save(self,sess):
        print("Saving model...")
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
        print("Model saved")

    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")
