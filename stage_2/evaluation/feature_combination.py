import os
import sys
import cv2
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from annoy import AnnoyIndex
from matplotlib import pyplot as plt

graph_path = "../image_net.pb"
ann_filename = "image_net_576_normal.ann"
ann_filename = 'image_net_only.ann'

graph_path_2 = "../YOLO.pb"
image_list = "image_list.txt"

IMG_HEIGHT=299
IMG_WIDTH=299

input_tensor_name = 'Mul:0'
feature_tensor_name = 'pool_3:0'

Feature_Dim = 2048
ann_tree_Num = 10

ann_generator =  AnnoyIndex(Feature_Dim)

def color_histogram(image):
    '''
    Extract the color feature from one image data
    input: image data
    return: color_feature (np.array)
    '''
    grid = 3
    lab = cv2.cvtColor(image,cv2.COLOR_RGB2LAB)
    h,w,_ = image.shape
    cell_size = h/grid
    color_image = []
    bins = [4,4,4]

    for i in range(grid):
        for j in range(grid):
            img_block = lab[i*cell_size:min((i+1)*cell_size,w-1),j*cell_size:min((j+1)*cell_size,h-1),:]
            histo_block = cv2.calcHist([img_block], [0, 1, 2],None, bins, [0, 256, 0, 256, 0, 256])
            if (i == int(grid/2)) and (j == int(grid/2)):
                color_image.append(np.reshape(2*histo_block,(64,)))
            else:
                #continue
                color_image.append(np.reshape(histo_block,(64,)))
    return np.reshape(color_image,(-1,))



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

        summary_writer = tf.train.SummaryWriter('./work/logs', graph=sess.graph)

        imge_index = 0
        with open(image_list) as f:
            for image_path in f:

                data_anchor = imresize(imread(image_path[:-1]),(IMG_HEIGHT,IMG_WIDTH))
                inputs = np.zeros((1,IMG_WIDTH, IMG_HEIGHT,3),dtype= "float32")
                inputs[0] = (data_anchor/255.0)*2.0-1.0

                input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)
                feature_tensor = sess.graph.get_tensor_by_name(feature_tensor_name)

                feature = sess.run([feature_tensor],{input_tensor:inputs})

                img = cv2.imread(image_path[:-1])
                img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                img_RGB = cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB) # BGR to RGB (OPENCV takes BRG as dafault)
                color_feature = np.array(color_histogram(img_RGB))  #The color feature

                color_feature = color_feature/(299.0*299.0)

                #print(feature[0][0][0][0].shape)
                #print(color_feature.shape)
                feature_combine = np.concatenate((feature[0][0][0][0], color_feature), axis=0)



                ann_generator.add_item(imge_index, feature_combine)
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
    index_test = 90
    data_anchor = imresize(imread(image_sets[index_test]),(IMG_HEIGHT,IMG_WIDTH))
    inputs = np.zeros((1,IMG_WIDTH, IMG_HEIGHT,3),dtype= "float32")
    inputs[0] = (data_anchor/255.0)*2.0-1.0

    input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)
    feature_tensor = sess.graph.get_tensor_by_name(feature_tensor_name)

    feature = sess.run([feature_tensor],{input_tensor:inputs})

    img = cv2.imread(image_sets[index_test])
    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img_RGB = cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB) # BGR to RGB (OPENCV takes BRG as dafault)
    color_feature = np.array(color_histogram(img_RGB))  #The color feature
    color_feature = color_feature/(299.0*299.0)
    feature_combine = np.concatenate((feature[0][0][0][0], color_feature), axis=0)
    similar_image_index = np.array((ann_loader.get_nns_by_vector(feature[0][0][0][0], target_num)))

    plt.subplot(3,5,1),plt.imshow(data_anchor),plt.title("Original_Image")

    for i in range(10):
        similar_image_name = image_sets[similar_image_index[i]]
        similar_image = imresize(imread(similar_image_name),(IMG_HEIGHT,IMG_WIDTH))
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
