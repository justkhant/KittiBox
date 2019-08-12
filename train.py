#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the TensorDetect model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys

# configure logging
if 'TV_IS_DEV' in os.environ and os.environ['TV_IS_DEV']:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)
else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, 'incl')
sys.path.insert(1, 'inputs')

import tensorvision.train as train
import tensorvision.utils as utils

flags.DEFINE_string('name', None,
                    'Append a name Tag to run.')

flags.DEFINE_string('project', None,
                    'Append a name Tag to run.')

flags.DEFINE_string('hypes', 'hypes/kittiBox.json',
                    'File storing model parameters.')

tf.app.flags.DEFINE_boolean(
    'save', True, ('Whether to save the run. In case --nosave (default) '
                   'output will be saved to the folder TV_DIR_RUNS/debug, '
                   'hence it will get overwritten by further runs.'))

flags.DEFINE_string('input_type', 'COLOR',
                    'Type of input training data')

def main(_):
    utils.set_gpus_to_use()

    try:
        import tensorvision.train
    except ImportError:
        logging.error("Could not import the submodules.")
        logging.error("Please execute:"
                      "'git submodule update --init --recursive'")
        exit(1)

    with open(tf.app.flags.FLAGS.hypes, 'r') as f:
        logging.info("f: %s", f)
        hypes = json.load(f)

    hypes["input_type"] = tf.app.flags.FLAGS.input_type
    if (hypes["input_type"] == 'COLOR'):
        hypes["input_channels"] = 3
        
    elif (hypes["input_type"] == 'GRAYSCALE'):
        hypes["input_channels"] = 1
        hypes["dirs"]["output_dir"] = 'RUNS/grayscale_box' 
    elif (hypes["input_type"] == 'EVENT'):
        hypes["input_channels"] = 9
        hypes["input_file"] = '../inputs/event_data_loader.py'
        hypes["data"]["train_file"] = 'data/event_train.txt'
        hypes["dirs"]["output_dir"] = 'RUNS/events_box'
        
    else:
        logging.error("data_type {} not supported.".format(hypes["input_type"]))
        exit(1)
    
    utils.load_plugins()
    
    if 'TV_DIR_RUNS' in os.environ:
        os.environ['TV_DIR_RUNS'] = os.path.join(os.environ['TV_DIR_RUNS'],
                                                 'KittiBox')
    utils.set_dirs(hypes, tf.app.flags.FLAGS.hypes)

    utils._add_paths_to_sys(hypes)

    logging.info("Initialize training folder")
    train.initialize_training_folder(hypes)
    train.maybe_download_and_extract(hypes)
    logging.info("Start training")
    train.do_training(hypes)


if __name__ == '__main__':
    tf.app.run()
