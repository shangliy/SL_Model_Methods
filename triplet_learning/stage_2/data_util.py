#!/usr/bin/python
#author: Shanglin Yang(kudoysl@gmail.com)

import os
import json
import numpy as np
import random
from scipy.misc import imread, imresize
import itertools

def parseJSON(filename):
    filename = os.path.realpath(filename)
    name, ext = os.path.splitext(filename)
    assert ext == '.json'

    annotations = []
    with open(filename, 'r') as f:
        jdoc = json.load(f)

    for annotation in jdoc:
        anno = {}
        anno["anchor"] = annotation["anchor"]
        anno["anchor_class"] = annotation["anchor_class"]
        anno["positive"] = annotation["positive"]
        anno["positive_class"] = annotation["positive_class"]
        anno["negative"] = annotation["negative"]
        anno["negative_class"] = annotation["negative_class"]
        annotations.append(anno)

    return annotations

def load_data(Data_file,n_labels,IMG_HEIGHT=299,IMG_WEIGHT=299):
    annolist = parseJSON(Data_file)

    anchor_name = []
    anchor_labels = []
    positive_name = []
    positive_labels = []
    negative_name = []
    negative_labels = []

    for anno in annolist:
        #anno.imageName = os.path.join(os.path.dirname(os.path.realpath(idlfile)), anno.imageName)
        anchor_name.append(anno["anchor"])
        anchor_labels.append(anno["anchor_class"])
        positive_name.append(anno["positive"])
        positive_labels.append(anno["positive_class"])
        negative_name.append(anno["negative"])
        negative_labels.append(anno["negative_class"])
    
    random.seed(0)
    
    for epoch in itertools.count():
    
        # Picking a positive pair
        data_anchor = imresize(imread(anchor_name[epoch]),(IMG_HEIGHT,IMG_WEIGHT))
        data_positive = imresize(imread(positive_name[epoch]),(IMG_HEIGHT,IMG_WEIGHT))

        # Picking a negative sample
        data_negative = imresize(imread(negative_name[epoch]),(IMG_HEIGHT,IMG_WEIGHT))

        label_anchor = np.reshape(np.array(anchor_labels[epoch]),[1,])
        label_positive = np.reshape(np.array(positive_labels[epoch]),[1,])
        label_negative = np.reshape(np.array(negative_labels[epoch]),[1,])
        
        yield {"train_anchor": data_anchor,"train_positive": data_positive,"train_negative": data_negative,\
                "label_anchor": label_anchor, "label_positive": label_positive, "label_negative":label_negative}


