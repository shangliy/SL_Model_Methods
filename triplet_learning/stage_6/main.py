"""
This is the basic main scripts for training
"""
import os
import argparse
import tensorflow as tf

from utils.dirs import create_dirs
from utils.configs import process_config
from models.tripletModel import tripletModel
from data_loader.dataGenerator import DataGenerator
from trainer.trainer import Trainer
#from utils.logger import Logger



def main():
    # capture the config from config files
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='configs',
                        help="Experiment directory containing params.json")


    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')

    config = process_config(json_path)

    # create the log directory
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create data generator
    data_gen = DataGenerator(config)


    # create the model
    model = tripletModel(config)

    # create the tensorboard logger
    #logger = Logger(sees, config)
    # create trainer and pass all previous components to it
    trainer = Trainer(sess, model, data_gen, config)
    # load model if exists
    #model.load(sess)
    # train model
    trainer.train()



if __name__ == '__main__':
        main()
