from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from scipy.misc import imread, imresize

import os
import numpy as np

import pylab as Plot
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Only one image as input
'''
img_path = 'dress.jpg'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
'''

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

def feature_extract(X_val):
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=True)
    model = Model(input=base_model.input, output=base_model.get_layer('flatten').output)
    model.load_weights("./model_weights",by_name=True)

    pool_features = model.predict(X_val)
    return pool_features
    

def csv_save(data,file_name):
    import csv
    csv_file = data
    with open(file_name, "wb") as ofile:
        writer = csv.writer(ofile)
        for row in csv_file:
            writer.writerow(row)

def csv_load(file_name):
    from numpy import genfromtxt
    return genfromtxt(file_name,delimiter=',')

def imscatter(x, y, images, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    
    x, y = np.atleast_1d(x, y)
    artists = []
    count = 0
    for x0, y0 in zip(x, y):
        try:
            image = plt.imread(images[count])
            count += 1
        except TypeError:
        # Likely already an array...
            pass
        im = OffsetImage(image, zoom=zoom)
        
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

def tsne_visul(X, image_set):
    import tsne
    Y = tsne.tsne(X, 2, 2048, 20.0);
    fig, ax = plt.subplots()
    imscatter(Y[:,0], Y[:,1], image_set, zoom=0.05, ax=ax)
    ax.scatter(Y[:,0], Y[:,1])
    plt.savefig('destination_path.eps', format='eps', dpi=1000)
    
def main():
    
    list_path = "./image_list.txt"
    csv_name = "bottleneck_test.csv"
    
    raw_data,image_set = load_image(list_path)

    if not os.path.isfile(csv_name):
        features = feature_extract(raw_data)
        csv_save(features, csv_name) 
    else:
        features = csv_load(csv_name)
        
    print("feature shapes: ",features.shape)
    
    tsne_visul(features, image_set)

    

if __name__ == "__main__":
    main()

