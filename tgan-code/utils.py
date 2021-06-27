import math
import tensorflow as tf
from tensorflow.keras.layers import concatenate
import random
import os
import scipy
import imageio
import numpy as np

import warnings
warnings.filterwarnings("ignore")

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

# def create_data_subsets(path=None, percent_train, percent_label):
#     (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.cifar10.load_data() #cifar10.load_data()
#     train_data, test_data = color_preprocessing(train_data, test_data) # pre-processing

#     criteria = n//10
#     input_dict, labelled_x, labelled_y, unlabelled_x, unlabelled_y = defaultdict(int), list(), list(), list(), list()

#     for image, label in zip(train_data,train_labels) :
#         if input_dict[int(label)] != criteria :
#             input_dict[int(label)] += 1
#             labelled_x.append(image)
#             labelled_y.append(label)

#         unlabelled_x.append(image)
#         unlabelled_y.append(label)


#     labelled_x = np.asarray(labelled_x)
#     labelled_y = np.asarray(labelled_y)
#     unlabelled_x = np.asarray(unlabelled_x)
#     unlabelled_y = np.asarray(unlabelled_y)

#     print("labelled data:", np.shape(labelled_x), np.shape(labelled_y))
#     print("unlabelled data :", np.shape(unlabelled_x), np.shape(unlabelled_y))
#     print("Test data :", np.shape(test_data), np.shape(test_labels))
#     print("======Load finished======")

#     print("======Shuffling data======")
#     indices = np.random.permutation(len(labelled_x))
#     labelled_x = labelled_x[indices]
#     labelled_y = labelled_y[indices]

#     indices = np.random.permutation(len(unlabelled_x))
#     unlabelled_x = unlabelled_x[indices]
#     unlabelled_y = unlabelled_y[indices]

#     print("======Prepare Finished======")


#     labelled_y_vec = np.zeros((len(labelled_y), 10), dtype=np.float)
#     for i, label in enumerate(labelled_y) :
#         labelled_y_vec[i, labelled_y[i]] = 1.0

#     unlabelled_y_vec = np.zeros((len(unlabelled_y), 10), dtype=np.float)
#     for i, label in enumerate(unlabelled_y) :
#         unlabelled_y_vec[i, unlabelled_y[i]] = 1.0

#     test_labels_vec = np.zeros((len(test_labels), 10), dtype=np.float)
#     for i, label in enumerate(test_labels) :
#         test_labels_vec[i, test_labels[i]] = 1.0


#     return labelled_x, labelled_y_vec, unlabelled_x, unlabelled_y_vec, test_data, test_labels_vec
    
#     return labelled_list, unlabelled_list, test_list

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


