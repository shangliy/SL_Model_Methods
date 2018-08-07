import tensorflow as tf
from tqdm import tqdm
import tensorflow.contrib.slim as slim


class Trainer():
    def __init__(self, sess, model, data_loader, config,  is_training=True):
        """Initilization of train

        Arguments:
            sess {[type]} -- [description]
            model {[type]} -- [description]
            config {[type]} -- [description]
            logger {[type]} -- [description]
            data_loader {[type]} -- [description]
        """

        self.sess = sess
        self.model = model
        self.config = config
        self.data_loader = data_loader
        self.cur_iterration = 0
        self.train_writer = tf.summary.FileWriter( self.config.summary_dir, sess.graph)

        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        if self.config.load:
            self.model.load(self.sess)
        else:

            saver_mobile = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='MobilenetV2'))
            checkpoint_name = config.checkpoint_dir + 'mobilenet_v2_1.0_224' #@param
            checkpoint = checkpoint_name + '.ckpt'
            saver_mobile.restore(sess,  checkpoint)


    def train(self):
        """
        This is the main loop of training
        Looping on the epochs
        :return:
        """

        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch(cur_epoch)
            self.model.global_step_assign_op.eval(session=self.sess, feed_dict={
                    self.model.global_step_input: self.model.global_step_tensor.eval(self.sess) + 1})


    def train_epoch(self, epoch=None):
        """
        Train one epoch
        :param epoch: cur epoch number
        :return:
        """

        # initialize tqdm
        tt = tqdm(range(self.config.num_iter_per_epoch),
                  desc="epoch-{}-".format(epoch))

        # Iterate over batches
        for cur_it in tt:
            # One Train step on the current batch
            loss, summary = self.train_step()

            self.train_writer.add_summary(summary, self.cur_iterration)
            self.cur_iterration += 1

            #print("""Epoch-{%s}  loss:{%.4f}"""%(epoch, loss))

        self.model.save(self.sess)



        tt.close()

    def train_step(self):
        """
        Run the session of train_step in tensorflow
        also get the loss & acc of that minibatch.
        :return: (loss, acc) tuple of some metrics to be used in summaries
        """
        batch_images = next(self.data_loader.next_batch())
        _, loss, summary, ea = self.sess.run([self.model.train_op, self.model.total_loss, self.model.merged, self.model.euclidean_a_p],
                                     feed_dict={self.model.input: batch_images, self.model.is_training: True})
        
        return loss, summary
