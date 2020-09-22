import math
import tensorflow as tf
from tensorflow.keras.layers import concatenate
import random
import os
import scipy
import imageio
import numpy as np

def rampup(epoch):
    if epoch < 80:
        p = max(0.0, float(epoch)) / float(80)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0

def conv_concat(x, y):
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()

    return concatenate([x, y * tf.ones([x_shapes[1], x_shapes[2], y_shapes[3]])], axis=3)

def create_data_subsets(path, percent_train, percent_label):    
    all_files = os.listdir(path)
    random.shuffle(all_files)
    
    train_threshold = len(all_files)*percent_train//100
    label_threshold = len(all_files[:train_threshold])*percent_label//100
    train_list = all_files[:train_threshold]
    test_list = all_files[train_threshold:]
    labelled_list = train_list[:label_threshold]
    unlabelled_list = train_list[label_threshold:]
    
    return labelled_list, unlabelled_list, test_list

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return imageio.imwrite(path, image)

def inverse_transform(images):
    #return (images+1.)/2.
    return ((images + 1.) * 127.5).astype('uint8')
    
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')
        
def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


