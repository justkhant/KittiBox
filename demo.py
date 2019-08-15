#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Marvin Teichmann


"""
Detects Cars in an image using KittiBox.

Input: Image
Output: Image (with Cars plotted in Green)

Utilizes: Trained KittiBox weights. If no logdir is given,
pretrained weights will be downloaded and used.

Usage:
               
python demo.py --input_image data/images [or data/images/000000.png or hdf5_file] 
                [--output_dir /path/to/output_images] [--logdir /path/to/weights] 
                [--gpus GPUs_to_use] [--save_boxes_dir /path/to/bounding_boxes]
        

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys
import h5py
from skimage import exposure

import collections

# configure logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import scipy as scp
import scipy.misc
import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, 'incl')
from utils import train_utils as kittibox_utils

try:
    # Check whether setup was done correctly
    import tensorvision.utils as tv_utils
    import tensorvision.core as core
except ImportError:
    # You forgot to initialize submodules
    logging.error("Could not import the submodules.")
    logging.error("Please execute:"
                  "'git submodule update --init --recursive'")
    exit(1)


flags.DEFINE_string('logdir', None,
                    'Path to logdir.')
flags.DEFINE_string('input_image', None,
                    'Input image, directory containing images, or hdf5 containing images to apply KittiBox.')
flags.DEFINE_string('output_dir', None,
                    'dir to store output images')
flags.DEFINE_string('save_boxes_dir', None,
                    'Directory to save coordinates of bounding boxes.') 

default_run = 'KittiBox_pretrained'
weights_url = ("ftp://mi.eng.cam.ac.uk/"
               "pub/mttt2/models/KittiBox_pretrained.zip")


def maybe_download_and_extract(runs_dir):
    logdir = os.path.join(runs_dir, default_run)

    if os.path.exists(logdir):
        # weights are downloaded. Nothing to do
        return

    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)

    import zipfile
    download_name = tv_utils.download(weights_url, runs_dir)

    logging.info("Extracting KittiBox_pretrained.zip")

    zipfile.ZipFile(download_name, 'r').extractall(runs_dir)

    return

def image_transform(image, hypes):

    #slice off the hood
    image = image[0:198,:]
   
    #resize image
    original_height, _ = image.shape
    new_width = int(hypes["image_width"] * original_height / hypes["image_height"])

    image = scp.misc.imresize(image, (hypes["image_height"], new_width),
                            interp='cubic')

    image = np.expand_dims(image, axis=2)
    #expand to 3 dims if color
    if (hypes["input_type"] == 'COLOR'):
        image = np.tile(image, (1, 1, 3)) 
   
    #pad sides
    pad_width_1 = int((hypes["image_width"] - new_width) / 2)
    pad_width_2 = hypes["image_width"] - new_width - pad_width_1
    image = np.pad(image, [(0, 0), (pad_width_1, pad_width_2), (0, 0)], mode='constant')

    image = exposure.adjust_gamma(image, 0.5) 

    return image 

def main(_):
    tv_utils.set_gpus_to_use()

    if FLAGS.input_image is None:
        logging.error("No input_images were given.")
        logging.info(
            "Usage: python demo.py --input_image data/images [or data/images/000000.png or hdf5_file] "
            "[--output_dir /path/to/output_images] [--logdir /path/to/weights] "
            "[--gpus GPUs_to_use] [--save_boxes_dir /path/to/bounding_boxes]")
        exit(1)

    if FLAGS.logdir is None:
        # Download and use weights from the MultiNet Paper
        if 'TV_DIR_RUNS' in os.environ:
            runs_dir = os.path.join(os.environ['TV_DIR_RUNS'],
                                    'KittiBox')
        else:
            runs_dir = 'RUNS'
        maybe_download_and_extract(runs_dir)
        logdir = os.path.join(runs_dir, default_run)
    else:
        logging.info("Using weights found in {}".format(FLAGS.logdir))
        logdir = FLAGS.logdir

    # Loading hyperparameters from logdir
    hypes = tv_utils.load_hypes_from_logdir(logdir, base_path='hypes')

    logging.info("Hypes loaded successfully.")

    # Loading tv modules (encoder.py, decoder.py, eval.py) from logdir
    modules = tv_utils.load_modules_from_logdir(logdir)
    logging.info("Modules loaded successfully. Starting to build tf graph.")

    # Create tf graph and build module.
    with tf.Graph().as_default():
        # Create placeholder for input
        image_pl = tf.placeholder(tf.float32)
        image = tf.expand_dims(image_pl, 0)
       
        # build Tensorflow graph using the model from logdir
        prediction = core.build_inference_graph(hypes, modules,
                                                image=image)

        logging.info("Graph build successfully.")

        # Create a session for running Ops on the Graph.
        sess = tf.Session()
        saver = tf.train.Saver()

        # Load weights from logdir
        core.load_weights(logdir, sess, saver)

        logging.info("Weights loaded successfully.")


    #Set images to evaluate    
    image_input = FLAGS.input_image 
    if (os.path.isdir(image_input)):
        if (hypes["input_type"] == 'GRAYSCALE'):
            logging.error("RGB images cannot be evaluated with network for grayscale") 
            exit(1)
        elif (hypes["input_type"] == 'EVENT'):
            logging.error("RGB images cannot be evaluated with network for events")
            exit(1)
        input_images = os.listdir(image_input)
        input_images = [os.path.join(image_input, x) for x in input_images]
    else:
        if (image_input.endswith(".hdf5")):
            hdf5_file = h5py.File(image_input, "r")
            input_images = hdf5_file["davis"]["left"]["image_raw"] 
            input_images = [input_images[i] for i in range(len(input_images)) if i % 60 == 0]  
            if (hypes["input_type"] == 'COLOR'):
                logging.info("WARNING:")
                logging.info("Evaluating grayscale images as RGB.")
        else:
            if (hypes["input_type"] == 'GRAYSCALE'):
                logging.error("RGB image cannot be evaluated with network for grayscale") 
                exit(1)
            elif (hypes["input_type"] == 'EVENT'):
                logging.error("RGB image cannot be evaluated with network for events")
                exit(1)
            input_images = [image_input] 


    #Evaluate images
    for i, input_image in enumerate(input_images):

        if not isinstance(input_image, basestring):
            image = image_transform(input_image, hypes)
            input_image = "image{:04}".format(i)

        elif (input_image.endswith(".png")):

            # Load and resize input image
            image = scp.misc.imread(input_image)
            image = scp.misc.imresize(image, (hypes["image_height"],
                                              hypes["image_width"]),
                                      interp='cubic')
    
        else:
            logging.info("{} is not in .png format. Edit in demo.py to accept other input file types".format(input_image))
            logging.info("")
            exit(1)
            
        logging.info("Starting inference using {} as input".format(input_image))
        
        feed = {image_pl: image}
        
        # Run KittiBox model on image
        pred_boxes = prediction['pred_boxes_new']
        pred_confidences = prediction['pred_confidences']
        (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes,
                                                        pred_confidences],
                                                        feed_dict=feed)
        if (image.shape[-1] == 1):
            image = np.tile(image, (1, 1, 3)) 
            
        # Apply non-maximal suppression
        # and draw predictions on the image   
        output_image, rectangles = kittibox_utils.add_rectangles(
                hypes, image, np_pred_confidences,
                np_pred_boxes, show_removed=False,
                use_stitching=True, rnn_len=1,
                min_conf=0.5, tau=hypes['tau'], color_acc=(0, 255, 0))
        
        threshold = 0
        accepted_predictions = []
        
        # removing predictions <= threshold
        for rect in rectangles:
            if rect.score >= threshold:
                accepted_predictions.append(rect) 
            
        print('')
        logging.info("{} Cars detected".format(len(accepted_predictions)))

        boxes_coords = []
            
        # Printing coordinates of predicted rects.
        for i, rect in enumerate(accepted_predictions):
            logging.info("")
            logging.info("Coordinates of Box {}".format(i))
            logging.info("    x1: {}".format(rect.x1))
            logging.info("    x2: {}".format(rect.x2))
            logging.info("    y1: {}".format(rect.y1))
            logging.info("    y2: {}".format(rect.y2))
            logging.info("    Confidence: {}".format(rect.score))
                
            boxes_coords.append("Car 0.00 0 0.00 {} {} {} {} 0 0 0 0 0 0 0 {}\n".format(rect.x1, rect.y1, rect.x2, rect.y2, rect.score))  
                
        #save boxes    
        if FLAGS.save_boxes_dir is not None:
            boxes_filename = os.path.join(FLAGS.save_boxes_dir, input_image.split('/')[-1].split('.')[0] + '_boxes.txt') 
            boxes_file = open(boxes_filename, "w")
            boxes_file.writelines(boxes_coords)
            boxes_file.close()
            logging.info("") 
            logging.info("Boxes saved to {}".format(boxes_filename))
                
        # save Image
        if FLAGS.output_dir is not None:
            output_name = os.path.join(FLAGS.output_dir, input_image.split('/')[-1].split('.')[0] + '_rects.png') 
            scp.misc.imsave(output_name, output_image)
            logging.info("")
            logging.info("Output image saved to {}".format(output_name))

            logging.info("")

        
 #   logging.warning("Do NOT use this Code to evaluate multiple images.")

 #   logging.warning("Demo.py is **very slow** and designed "
 #                   "to be a tutorial to show how the KittiBox works.")
 #   logging.warning("")
 #   logging.warning("Please see this comment, if you like to apply demo.py to"
 #                   "multiple images see:")
 #   logging.warning("https://github.com/MarvinTeichmann/KittiBox/"
 #                   "issues/15#issuecomment-301800058")

if __name__ == '__main__':
    tf.app.run()
