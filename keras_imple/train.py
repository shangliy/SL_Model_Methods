from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from scipy.misc import imread, imresize
from keras.layers import merge, Convolution2D, MaxPooling2D, Input, Dense, Flatten, Reshape
from keras.engine.topology import Layer


import os
import numpy as np

import pylab as Plot
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

BATCH_SIZE = 3


class Triplet(Layer):
    """ Computes $x^T zI y rowwise for a batch"""
    def __init__(self, **kwargs):
        super(Triplet, self).__init__(**kwargs)

    def build(self, input_shape):
        #assert len(input_shape) == 3, "Input should be shape (batch_size, 3, embed_size)"
        embed_dim = input_shape[1] # 0 is the batch dim
        self.trainable_weights = []

    def call(self, tensor, mask=None):
        x = tensor[:,0:2048]
        y = tensor[:,2048:4096]
        z = tensor[:,4096:]
        triplet_loss = K.maximum(1.0
                     + K.sum(x * z, axis=-1, keepdims=True)
                     - K.sum(x * y, axis=-1, keepdims=True),
                     0.0)
        return triplet_loss

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 1)


def load_image(list_path):
    IMG_HEIGHT=299
    IMG_WEIGHT=299
    image_set = []
    label_set = []
    with open(list_path) as f:
        for image_path in f:
            image_set.append(image_path[:-1])
            label_name = image_path[image_path.rfind('/')+1:-7]
            label_set.append(label_name)

    image_num = len(image_set)
    print("image_num: ",image_num)
    X_val = np.zeros(shape=(image_num, 299, 299, 3))

    for i in range(image_num):
        image_path = image_set[i]
        data_anchor = imresize(imread(image_path),(IMG_HEIGHT,IMG_WEIGHT))
        X_val[i] = (data_anchor/255.0)*2.0-1.0
   
    return X_val,image_set

def model_train(X):
    # create the base pre-trained model

    input_tensor = Input(shape=(299, 299, 3))
    base_model = InceptionV3(input_tensor=input_tensor,weights='imagenet', include_top=True)
    inception_model = Model(input=base_model.input, output=base_model.get_layer('flatten').output)

    anchor = Input(shape=(299, 299, 3))
    positi = Input(shape=(299, 299, 3))
    negati = Input(shape=(299, 299, 3))

    fea_anchor = Reshape(target_shape=(2048,))(inception_model(anchor))
    pos_anchor = Reshape(target_shape=(2048,))(inception_model(positi))
    neg_anchor = Reshape(target_shape=(2048,))(inception_model(negati))

    for layer in inception_model.layers[:172]:
        layer.trainable = False
    for layer in inception_model.layers[172:]:
        layer.trainable = True

    embbedings = merge([fea_anchor, pos_anchor, neg_anchor], mode='concat')
    triplet_loss = Triplet()(embbedings)    
    
    feature_model = Model([anchor,positi,negati],triplet_loss)

    def identity_loss(y_true, y_pred):
         return K.mean(y_pred)

    feature_model.compile(optimizer='rmsprop', loss=identity_loss)
    y = np.zeros(shape = (X.shape[0],1))

    feature_model.fit([X, X, X], y ,nb_epoch = 5, batch_size=1) 

    feature_model.save_weights("./model_weights")

    from keras.utils.visualize_util import plot
    plot(feature_model, to_file='model.png')
    feature_model.save('my_model_new.h5')
    
    return feature_model
    
    
    



def main():
    
    list_path = "./image_list.txt"
    csv_name = "bottleneck.csv"
    

    raw_data,image_set = load_image(list_path)

    model_train(raw_data)    

if __name__ == "__main__":
    main()