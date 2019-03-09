import os
import sys
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from annoy import AnnoyIndex
from matplotlib import pyplot as plt

graph_path = "../triplet_1000.pb"
ann_filename = "triplet_1000.ann"
graph_path = "../triplet_1.pb"
ann_filename = "triplet_1.ann"

#graph_path = "../image_net.pb"
#ann_filename = "image_net.ann"

graph_path_2 = "../YOLO.pb"
image_list = "image_list.txt"

IMG_HEIGHT=299
IMG_WEIGHT=299
input_tensor_name = "fifo_queue_DequeueMany:0"
feature_tensor_name = "Triplet/logits/flatten/Reshape:0"

#input_tensor_name = 'Mul:0'
#feature_tensor_name = 'pool_3:0'

Feature_Dim = 2048
ann_tree_Num = 10

ann_generator =  AnnoyIndex(Feature_Dim)
######################################################################
# Unpersists graph from file
with tf.gfile.FastGFile(graph_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
######################################################################

# Do not use GPU when testing
config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

if (not os.path.isfile(ann_filename)):

    with tf.Session(config=config) as sess:
    #sess.graph.as_default()

        #summary_writer = tf.train.SummaryWriter('./work/logs', graph=sess.graph)
        imge_index = 0
        with open(image_list) as f:
            for image_path in f:

                data_anchor = imresize(imread(image_path[:-1]),(IMG_HEIGHT,IMG_WEIGHT))
                inputs = np.zeros((1,IMG_WEIGHT, IMG_HEIGHT,3),dtype= "float32")
                inputs[0] = (data_anchor/255.0)*2.0-1.0

                input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)
                feature_tensor = sess.graph.get_tensor_by_name(feature_tensor_name)

                feature = sess.run([feature_tensor],{input_tensor:inputs})
                #print(feature[0].shape)
                ann_generator.add_item(imge_index, feature[0][0])
                print(imge_index)
                imge_index += 1

    print(imge_index)
    ann_generator.build(ann_tree_Num) # 10 trees
    ann_generator.save(ann_filename)

with open(image_list, 'r') as img_ori_lists:
    image_sets = img_ori_lists.read().splitlines()

ann_loader = AnnoyIndex(Feature_Dim)
ann_loader.load(ann_filename)
target_num = 10

with tf.Session(config=config) as sess:

    data_anchor = imresize(imread(image_sets[97]),(IMG_HEIGHT,IMG_WEIGHT))
    inputs = np.zeros((1,IMG_WEIGHT, IMG_HEIGHT,3),dtype= "float32")
    inputs[0] = (data_anchor/255.0)*2.0-1.0

    input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)
    feature_tensor = sess.graph.get_tensor_by_name(feature_tensor_name)

    feature = sess.run([feature_tensor],{input_tensor:inputs})

    similar_image_index = np.array((ann_loader.get_nns_by_vector(feature[0][0], target_num)))

    plt.subplot(3,5,1),plt.imshow(data_anchor),plt.title("Original_Image")

    for i in range(10):
        similar_image_name = image_sets[similar_image_index[i]]
        similar_image = imresize(imread(similar_image_name),(IMG_HEIGHT,IMG_WEIGHT))
        plt.subplot(3,5,i+6),plt.imshow(similar_image),plt.title('similat_'+str(i+1))
    plt.show()







'''
with tf.gfile.FastGFile(graph_path_2, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
######################################################################
with tf.Session() as sess:
#sess.graph.as_default()
    summary_writer = tf.train.SummaryWriter('./work/logs', graph=sess.graph)

'''
