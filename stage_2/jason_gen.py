#!/usr/bin/python
#author: Shanglin Yang
"""
This Program is to generate the json file for triplet training
The output json contain "MAXIMUM_DATA" triplets, each triplet contains 6 parameter:

"anchor": the path of anchor image
"anchor_class": the class id of anchor image
"positive": the path of positive image
"positive_class": the class id of positive image
"negative": the path of negative image
"negative_class": the class id of negative image

INPUT:
"IMG_ROOT"
"OUT_FILE"
"MAXIMUM_DATA"

Note to learn:
os.listdir / os.isdir / os.path.join:   
    for item in os.listdir(IMG_ROOT) if os.path.isdir(os.path.join(IMG_ROOT, item))

no.random.choice
"""

import os
import json
import glob
import numpy as np
from scipy.misc import imread, imresize

IMG_ROOT = "/media/shangliy/Storage/Source/realreal_triplet/"
OUT_FILE = "./triplet_fine_data.json"
MAXIMUM_DATA = 10000

# List the class name
CLASS_NAME =  [ item for item in os.listdir(IMG_ROOT) if os.path.isdir(os.path.join(IMG_ROOT, item)) ]
CLASS_NUM = len(CLASS_NAME)

# Generate the image list for each class
image_list = []
for i in range(CLASS_NUM):
    #print(os.path.join(IMG_ROOT, CLASS_NAME[i]))
    image_list.append( glob.glob(os.path.join(IMG_ROOT, CLASS_NAME[i])+"/*.jpg") )
    print(CLASS_NAME[i])
    print(len(image_list[i]))

#The num of the triplets for training
num = 0
Data_Json = []

while(num < MAXIMUM_DATA):
    
    json_dict = {}
    # Getting a pair of clients
    index = np.random.choice(CLASS_NUM,2,replace=False)
    label_positive = index[0]
    label_negative = index[1]

    # Getting the indexes of the data from a particular client
    indexes = image_list[label_positive]
    np.random.shuffle(indexes)

    # Picking a positive pair
    data_anchor_name = indexes[0]
    data_positive_name = indexes[1]
    #data_anchor = imread(indexes[0])
    #data_positive = imread(indexes[1])
    json_dict["anchor"] = data_anchor_name
    json_dict["anchor_class"] = label_positive
    json_dict["positive"] = data_positive_name
    json_dict["positive_class"] = label_positive

    # Picking a negative sample
    indexes = image_list[label_negative]
    np.random.shuffle(indexes)

    data_negative_name = indexes[0]
    #data_negative = imread(indexes[0])
    json_dict["negative"] = data_negative_name
    json_dict["negative_class"] = label_negative

    Data_Json.append(json_dict)
    num = num + 1

outfile = open(OUT_FILE, 'w')
json.dump(Data_Json, outfile, sort_keys = True, indent = 4)
outfile.close()









                
                

