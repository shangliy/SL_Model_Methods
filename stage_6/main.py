"""
This is the basic main scripts for training
"""

import argparse
import tensorflow as tf

from utils.dirs import create_dirs
from urils.configs import process_config
from models.tripletModel import tripletModel
from utils.logger import Logger
from utils.utils import get_args


def main():
    # capture the config from config files
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='experiments/base_model',
                        help="Experiment directory containing params.json")
    parser.add_argument('--data_dir', default='data/mnist',
                        help="Directory containing the dataset")

    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')

    config = process_config(json_path)
    
    # create the log directory
    create_dirs([config.summary_sir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create data generator
    data_gen = dataGenerator(config)

    # create the model
    model = tripletModel(config)
    # create the tensorboard logger
    #logger = Logger(sees, config)
    # create trainer and pass all previous components to it
    trainer = Trainer(sess, model, data, config, logger)
    # load model if exists
    model.load(sess)
    # train model
    trainer.train()

    if __name__ == '__main__':
        main()
