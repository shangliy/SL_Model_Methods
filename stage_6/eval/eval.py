"""
This is the basic main scripts for training
"""
import os
import json
import argparse
import numpy as np
import tensorflow as tf

from annoy import AnnoyIndex




from utils.dirs import create_dirs
from utils.configs import process_config
from models.tripletModel import tripletModel
from data_loader.dataGenerator import DataGenerator
from trainer.trainer import Trainer

def fea_extract(model, sess, image_path, config):
    image_data = tf.keras.preprocessing.image.load_img(image_path, target_size=(config.input_size, config.input_size))
    x_ = np.array(image_data).astype(np.float32) / 128.  - 1
    x_ = np.expand_dims(x_, axis=0)
    fea = sess.run([model.visualembedding],
                                 feed_dict={model.input: x_, model.is_training: False})
    return fea[0][0][0][0]

# image candidates
# capture the config from config files
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='configs',
                    help="Experiment directory containing params.json")


args = parser.parse_args()
json_path = os.path.join(args.model_dir, 'params.json')

config = process_config(json_path)
with open('data/recom_res.json') as f:
    triplet_set_list = json.load(f)

model = tripletModel(config)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
model.load(sess)

f = config.embedding_dim
  # Length of item vector that will be indexed
for skuid in triplet_set_list:
    sku_candidates = set()
    candidate_list = []
    t = AnnoyIndex(f,metric='euclidean')
    for key in triplet_set_list[skuid]:
        for candiate_skuid in triplet_set_list[skuid][key]:
            sku_candidates.add(candiate_skuid)

    data_dir = config.dataDir
    X = np.zeros((len(sku_candidates), config.embedding_dim))
    for idx, canditates in enumerate(sku_candidates):
        image_path = os.path.join(data_dir, "%s.jpg"%(canditates))

        v_ = fea_extract(model, sess,image_path, config)
        t.add_item(idx, v_)
        candidate_list.append(canditates)

    t.build(10) # 10 trees

    image_path = os.path.join(data_dir, "%s.jpg"%(skuid))
    v = fea_extract(model, sess,image_path, config)
    print(skuid)
    for s in (t.get_nns_by_vector(v, 3, search_k=-1, include_distances=False)):
        print(candidate_list[s])
    print('------------------------------------------------------')
