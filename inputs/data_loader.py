from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys
import random
from random import shuffle
import itertools
import pdb
import numpy as np

sys.path.insert(1, '../incl')

import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt

from utils.data_utils2 import annotation_to_h5
from utils.annolist import AnnotationLib2 as AnnoLib
from utils.rect import Rect

from collections import namedtuple
fake_anno = namedtuple('fake_anno_object', ['rects'])

import h5py
import gen_event_volume as gev
import numpy as np

def _projection(point, calib):
    point_r = np.reshape(point, (3, ))
    point_exp = np.reshape([point_r[0], point_r[1], point_r[2], 1], (4, 1))
    point_proj = np.dot(calib, point_exp)
    point_proj = point_proj[:2] / point_proj[2]
    return np.reshape(point_proj, (2, ))

def _vis(im_obj, corners):
    plt.figure(figsize=(12, 4))
    #plt.imshow(np.clip(im_obj, 0, 256)).astype(np.int32))
    plt.imshow(im_obj) 
    for r in corners:
        x1 = r[0]
        x2 = r[1]
        y1 = r[2]
        y2 = r[3]
        plt.plot([x1, x2, x2, x1, x1], 
                 [y1, y1, y2, y2, y1])
        #bottom_proj = _projection([r.x_3d, r.y_3d, r.z_3d], r.calib)
        #plt.scatter(bottom_proj[0], bottom_proj[1])
    plt.show() 
    #plt.savefig('/home/khantk/MonoGRNet/{}'.format(index))
    #plt.close()
    return

def read_kitti_anno(label_file, calib_file, detect_truck):
    """ Reads a kitti annotation file.

    Args:
    label_file: Path to file

    Returns:
      Lists of rectangels: Cars and don't care area.
    """
    labels = [line.rstrip().split(' ') for line in open(label_file)]

    label_file_split = label_file.rstrip().split('/')
    index = label_file_split[-1].split('.')[0]
    #import pdb 
    #pdb.set_trace()
    calibs = [line.rstrip().split(' ') for line in open(calib_file)]
    assert calibs[2][0] == 'P2:'
    calib = np.reshape(calibs[2][1:], (3, 4)).astype(np.float32)
    calib_pinv = np.linalg.pinv(calib)
    rect_list = []
    for label in labels:
        if not (label[0] == 'Car' or label[0] == 'Van' or
                label[0] == 'Truck' or label[0] == 'DontCare'):
            continue
        notruck = not detect_truck
        if notruck and label[0] == 'Truck':
            continue
        if label[0] == 'DontCare':
            class_id = -1
        else:
            class_id = 1
        object_rect = AnnoLib.AnnoRect(
            x1=float(label[4]), y1=float(label[5]),
            x2=float(label[6]), y2=float(label[7]),
            height=float(label[8]), width=float(label[9]),
            length=float(label[10]), x=float(label[11]), 
            y=float(label[12]), z=float(label[13]), 
            alpha=float(label[14]), calib=calib, 
            calib_pinv=calib_pinv)
        assert object_rect.x1 < object_rect.x2
        assert object_rect.y1 < object_rect.y2
        object_rect.classID = class_id

        view_angle = np.arctan2(object_rect.z_3d, object_rect.x_3d)
        object_rect.alpha += view_angle - np.pi * 0.5

        rect_list.append(object_rect)
    return rect_list

def _rescale_boxes(current_shape, anno, target_height, target_width):
    x_scale = target_width / float(current_shape[1])
    y_scale = target_height / float(current_shape[0])
    z_3d_scale = ((x_scale**2 + y_scale**2)*0.5)**0.5
    for r in anno.rects:
        assert r.x1 < r.x2
        r.x1 *= x_scale
        r.x2 *= x_scale
        assert r.x1 < r.x2
        r.y1 *= y_scale
        r.y2 *= y_scale
        r.xy_scale = np.array([x_scale, y_scale], dtype=np.float32)
    return anno


def _generate_mask(hypes, ignore_rects):

    width = hypes["image_width"]
    height = hypes["image_height"]
    grid_width = hypes["grid_width"]
    grid_height = hypes["grid_height"]

    mask = np.ones([grid_height, grid_width])

    if not hypes['use_mask']:
        return mask

    for rect in ignore_rects:
        left = int((rect.x1+2)/width*grid_width)
        right = int((rect.x2-2)/width*grid_width)
        top = int((rect.y1+2)/height*grid_height)
        bottom = int((rect.y2-2)/height*grid_height)
        for x in range(left, right+1):
            for y in range(top, bottom+1):
                mask[y, x] = 0

    return mask

def read_hdf5_file(hdf5_file, hypes):

    data = h5py.File(hdf5_file, "r")
    events = data["davis"]["events"].value
    event_mask = np.logical_and(events[:, 0] < hypes["image_width"] - 1,
                                events[:, 1] < hypes["image_height"] - 1)
    events = events[event_mask, :]

    events = events[np.newaxis, ...].astype(np.float32)
    
    return events

def read_anno(gt_image_file, calib_file, hypes): 
    rect_list = read_kitti_anno(gt_image_file, calib_file,
                                detect_truck=hypes['detect_truck'])
    anno = AnnoLib.Annotation()
    anno.rects = rect_list

    original_shape = (375, 1242) 
            
    anno = _rescale_boxes(original_shape, anno,
                         hypes["image_height"],
                         hypes["image_width"])

    pos_list = [rect for rect in anno.rects if rect.classID == 1]
    pos_anno = fake_anno(pos_list)
    corners = np.asarray([(rect.x1, rect.x2, rect.y1, rect.y2) for rect in pos_list])

    # boxes: [1, grid_height*grid_width, 11, max_len, 1]
    # for each cell, this array contains the ground truth boxes around it (within focus area, defined by center distance)
    # confs: [1, grid_height*grid_width, 1, max_len, 1]
    # record the valid boxes, since max_len is greater than the number of ground truth boxes
    boxes, confs, calib, calib_pinv,  xy_scale = annotation_to_h5(hypes,
                                                                  pos_anno,
                                                                  hypes["grid_width"],
                                                                  hypes["grid_height"],
                                                                  hypes["rnn_len"]) 
    # masks are zero in "Don't care" area 
    mask_list = [rect for rect in anno.rects if rect.classID == -1]
    mask = _generate_mask(hypes, mask_list)
    
    boxes = boxes.reshape([hypes["grid_height"],
                           hypes["grid_width"], 11])
    confs = confs.reshape(hypes["grid_height"], hypes["grid_width"])
    calib = calib.reshape(hypes["grid_height"], 
                          hypes["grid_width"], 3, 4)
    xy_scale = xy_scale.reshape(hypes["grid_height"], 
                                hypes["grid_width"], 2)
    calib_pinv = calib_pinv.reshape(hypes['grid_height'], 
                                    hypes['grid_width'], 4, 3)
    
    return boxes.astype(np.float32), confs.astype(np.float32), \
        calib.astype(np.float32), calib_pinv.astype(np.float32), \
        xy_scale.astype(np.float32), mask.astype(np.float32), \
        corners.astype(np.float32)

    
def load_data(hypes, hdf5_file, gt_image_file, calib_file):
    read_hdf5_file_tf = lambda hdf5: read_hdf5_file(hdf5, hypes) 

    [events] = tf.py_func(read_hdf5_file_tf, [hdf5_file], [tf.float32])
    events.set_shape((1, None, 4))

    event_volume, _ = gev.gen_interpolated_event_volume(events,
                                              (1, hypes["image_height"],
                                               hypes["image_width"], 9)) 

    event_volume = tf.squeeze(event_volume)

    read_anno_tf = lambda gt_file, ca_file: read_anno(gt_file, ca_file, hypes) 
    
    boxes, confs, calib, calib_pinv, xy_scale, mask, corners = tf.py_func(
        read_anno_tf, [gt_image_file, calib_file],
        [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])  

    return event_volume, (confs, boxes, mask, calib, calib_pinv, xy_scale)


def create_dataset(kitti_txt, hypes, random_shuffle1=True):
    hdf5_files = []
    calib_files = []
    gt_image_files = []
    image_files = []
    
    base_path = os.path.realpath(os.path.dirname(kitti_txt))
    files = [line.rstrip() for line in open(kitti_txt)]
    if random_shuffle1:
        random.shuffle(files) 
    for file in files:
        hdf5_file, gt_image_file = file.split(" ")
        hdf5_file_split = hdf5_file.split('/')
        index = hdf5_file_split[-1].split('_')[0]

        calib_file = os.path.join(base_path, hdf5_file_split[0], 'calib', index + '.txt')
        assert os.path.exists(calib_file), \
            "File does not exist: %s" % calib_file
        hdf5_file = os.path.join(base_path, hdf5_file)
        assert os.path.exists(hdf5_file), \
            "File does not exist: %s" % hdf5_file
        gt_image_file = os.path.join(base_path, gt_image_file)
        assert os.path.exists(gt_image_file), \
            "File does not exist: %s" % gt_image_file
        #image_file = os.path.join(base_path, hdf5_file_split[0], 'image_2', index + '.png')
        #assert os.path.exists(image_file), \
            #"File does not exist: $s" % image_file
        
        hdf5_files.append(hdf5_file)
        calib_files.append(calib_file)
        gt_image_files.append(gt_image_file)
        #image_files.append(image_file) 
        
    hdf5_files_t = tf.constant(hdf5_files)
    calib_files_t = tf.constant(calib_files)
    gt_image_files_t = tf.constant(gt_image_files) 
    #image_files_t = tf.constant(image_files)
    dataset = tf.data.Dataset.from_tensor_slices((hdf5_files_t,
                                                  gt_image_files_t,
                                                  calib_files_t))

    hypes["rnn_len"] = 1
    dataset = dataset.map(lambda hdf5, gt, calib: (load_data(hypes, hdf5, gt, calib)),
                          num_parallel_calls=hypes["num_parallel_map_calls"])

    batched_dataset = dataset.batch(hypes["batch_size"])
    batched_dataset = batched_dataset.prefetch(buffer_size=hypes["batch_size"])  
    iterator = batched_dataset.make_one_shot_iterator()
    next_element = iterator.get_next() 

    return next_element 

def main():
    np.set_printoptions(threshold=np.inf) 

    hypes_file = "/home/khantk/MonoGRNet/hypes/kittiBox.json"
    with open(hypes_file, "r") as f:
        hypes = json.load(f)  
    
    next_element = create_dataset("/home/khantk/MonoGRNet/data/KittiBox/train.txt", hypes) 
            
    sess = tf.Session()

    for i in range(0, 10): 
        events_volume, labels = sess.run(next_element)
        #first_volume = np.sum(events_volume, 3)[0]   

        #color_img = plt.imread(img_path[0])
        #import cv2
        #color_img = cv2.resize(color_img, (861, 260)) 
        
        #_vis(color_img, corners) 
    
    sess.close() 

if __name__ == '__main__':
    main()
