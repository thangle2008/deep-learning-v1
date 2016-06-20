import os
import re

import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from skimage.color import rgb2gray

import random
import cPickle
import gzip

from scipy.misc import imread, imresize, imrotate
from scipy.ndimage.interpolation import zoom, shift

SEED1 = 42
SEED2 = 43

def load_image(folder, dim=28, expand_train=False, mode="L", add_gray=False):
    print "Loading data"
    images = []
    categories = []
    c = 0

    # load the images in grayscale and resize
    for root, dirnames, filenames in os.walk(folder):  
        if root == folder:
            continue
        print root
        for filename in filenames:
            if re.search("\.(jpg|png)$", filename):
                filepath = os.path.join(root, filename)
                image = imread(filepath, mode=mode)
                gray_img = None
                if mode == "L":
                    image = imresize(image, (dim, dim))
                    image = np.array([image])
                if mode == "RGB":
                    image = imresize(image, (dim, dim))
                    # normalizing
                    image = image / 255.0

                    if add_gray:
                        gray_img = rgb2gray(image)
                        gray_img = np.array([gray_img, gray_img, gray_img])
                    image = image.swapaxes(0, 2).swapaxes(1, 2)

                images.append(image)
                categories.append(c)
                if add_gray:
                    images.append(gray_img)
                    categories.append(c) 
        c += 1
    
    images = np.asarray(images, dtype = np.float32)
    categories = np.array(categories)
    
    # stratified shuffle and split the data set
    sss = StratifiedShuffleSplit(categories, 1, test_size=0.4,
                        random_state=SEED1)

    train_x, train_y, test_val_x, test_val_y = None, None, None, None

    for train_index, test_index in sss:
        train_x, test_val_x = images[train_index], images[test_index]
        train_y, test_val_y = categories[train_index], categories[test_index]

    # continue to split between val and test sets
    sss = StratifiedShuffleSplit(test_val_y, 1, test_size = 0.5, 
                        random_state=SEED2)
    
    val_x, val_y, test_x, test_y = None, None, None, None
    for val_index, test_index in sss:
        val_x, test_x = test_val_x[val_index], test_val_x[test_index]
        val_y, test_y = test_val_y[val_index], test_val_y[test_index]

    # expand the training data if desired
    if expand_train:
        train_x, train_y = expand_data_set(list(train_x), list(train_y))

    print np.array(train_x).shape
    train_data = (np.array(train_x), np.array(train_y))
    val_data = (val_x, val_y)
    test_data = (test_x, test_y)

    return train_data, val_data, test_data

def save_image(data, filename="data_set.pkl.gz"):
    print "Saving data"
    f = gzip.open(filename, "w")
    cPickle.dump(data, f)
    f.close() 

def expand_data_set(imgs, labels):
    print "expanding ..."
    num_channels = len(imgs[0].shape)
    def augment(a_imgs, a_labels, fn_list, param):
        new_imgs = []
        new_labels = []
        for i in range(len(a_imgs)):
            img = a_imgs[i]
            if num_channels == 3:
                img = a_imgs[i].swapaxes(1, 2).swapaxes(0, 2)
            label = a_labels[i]
            for f, p in zip(fn_list, param):
                new_img = f(img, p) if p else f(img)
                if num_channels == 3:
                    new_img = new_img.swapaxes(0, 2).swapaxes(1, 2)
                new_imgs.append(new_img)
                new_labels.append(label)
        return (a_imgs + new_imgs, a_labels + new_labels)
   
    # rotating 90, 180 and 270 degree
    new_imgs, new_labels = augment(imgs, labels, [imrotate, imrotate, imrotate], [90, 180, 270])

    # flip up-down, left-right
    #new_imgs, new_labels = augment(new_imgs, new_labels, [np.flipud, np.fliplr], [None, None])
    
    # shift 1 pixel each direction
    # if num_channels == 3:
    #    new_imgs, new_labels = augment(new_imgs, new_labels, [shift, shift, shift, shift], 
    #                               [(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 1.0, 0.0)])  
    #else:
    #    new_imgs, new_labels = augment(new_imgs, new_labels, [shift, shift, shift, shift], 
    #                               [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)])  
    return np.array(new_imgs), np.array(new_labels)
