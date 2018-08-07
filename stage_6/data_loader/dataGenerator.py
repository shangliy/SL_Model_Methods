"""
This script is the data generator
"""
import json
import os
import tensorflow as tf
import numpy as np

class DataGenerator:
    """
    Data generator for training
    """
    def __init__(self, config):
        self.config = config
        self.dataDir = config.dataDir
        self.batch_size = config.batch_size
        self.X = np.zeros((self.batch_size * 3, config.input_size, config.input_size , 3))

        self.load_images()
    
    def load_images(self):
        with open(self.config.triJson) as f:
            self.triplet_set_list = json.load(f) 
        self.setNum = len(self.triplet_set_list)

    def next_batch(self):

        batch_idx = np.random.choice(self.setNum, self.batch_size)
        for id_index in range(self.batch_size):
            id_ = batch_idx[id_index]
            image_path_a = os.path.join(self.config.dataDir, self.triplet_set_list[id_]['a'])
            image_path_p = os.path.join(self.config.dataDir, self.triplet_set_list[id_]['p'])
            image_path_n = os.path.join(self.config.dataDir, self.triplet_set_list[id_]['n'])
            image_data_a = tf.keras.preprocessing.image.load_img(image_path_a, target_size=(self.config.input_size, self.config.input_size))
            image_data_p = tf.keras.preprocessing.image.load_img(image_path_p, target_size=(self.config.input_size, self.config.input_size))
            image_data_n = tf.keras.preprocessing.image.load_img(image_path_n, target_size=(self.config.input_size, self.config.input_size))
            self.X[id_index] = np.array(image_data_a).astype(np.float32) / 128.  - 1
            self.X[id_index + self.batch_size] = np.array(image_data_p).astype(np.float32) / 128.  - 1
            self.X[id_index + 2*self.batch_size] = np.array(image_data_n).astype(np.float32) / 128.  - 1
        
        yield self.X