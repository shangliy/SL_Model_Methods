"""
This script is the data generator
"""
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
    
    def load_images(self):
        with open(self.config.dataJson) as f:
            self.triplet_set_list = json.load(f) 
        self.setNum = len(self.triplet_set_list)

    def next_batch(self, batch_size):

        batch_idx = np.random.choice(self.setNum, batch_size)
        for id_ in batch_idx:
            image_path_a = os.path.join(self.config.dataDir, self.triplet_set_list[id_]['a'])
            image_path_p = os.path.join(self.config.dataDir, self.triplet_set_list[id_]['p'])
            image_path_n = os.path.join(self.config.dataDir, self.triplet_set_list[id_]['n'])
            image_data_a = tf.keras.preprocessing.image.load_img(image_path_a)
            image_data_p = tf.keras.preprocessing.image.load_img(image_path_p)
            image_data_n = tf.keras.preprocessing.image.load_img(image_path_n)
            self.X[id_] = np.array(image_data_a).astype(np.float32)
            self.X[id_ + batch_size] = np.array(image_data_p).astype(np.float32)
            self.X[id_ = 2*batch_size] = np.array(image_data_n).astype(np.float32)
        
        yield self.X