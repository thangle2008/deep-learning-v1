import os
import re

import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit

import random
import cPickle
import gzip

from scipy.misc import imread, imresize, imrotate
from scipy.ndimage.interpolation import zoom, shift

SEED = 42

def load_image(folder, dim=28, expand_train=False, mode="L"):
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
                if mode == "L":
                    image = imresize(image, (dim, dim))
                if mode == "RGB":
                    image = imresize(image, (dim, dim))
                    image = image.swapaxes(0, 2).swapaxes(1, 2)
                images.append(image)
                categories.append(c)
        c += 1
    
    images = np.asarray(images, dtype = np.float32)
    print images.shape
    categories = np.array(categories)
    
    # normalizing
    images = images / 255.0

    # stratified shuffle and split the data set
    sss = StratifiedShuffleSplit(categories, 1, test_size=0.4,
                        random_state=SEED)

    train_x, train_y, test_val_x, test_val_y = None, None, None, None

    for train_index, test_index in sss:
        train_x, test_val_x = images[train_index], images[test_index]
        train_y, test_val_y = categories[train_index], categories[test_index]

    num_test_val = test_val_x.shape[0]
    val_x, val_y = test_val_x[:num_test_val/2], test_val_y[:num_test_val/2]
    test_x, test_y = test_val_x[num_test_val/2:], test_val_y[num_test_val/2:]

    if expand_train:
        train_x, train_y = expand_data_set(list(train_x), list(train_y))

    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    test_data = (test_x, test_y)

    return train_data, val_data, test_data

def save_image(data, filename="data_set.pkl.gz"):
    f = gzip.open(filename, "w")
    cPickle.dump(data, f)
    f.close() 

def expand_data_set(imgs, labels):
    print "expanding ..."
    def augment(a_imgs, a_labels, fn_list, param):
        new_imgs = []
        new_labels = []
        for i in range(len(a_imgs)):
            img = a_imgs[i]
            label = a_labels[i]
            for f, p in zip(fn_list, param):
                new_img = f(img, p) if p else f(img)
                new_imgs.append(new_img)
                new_labels.append(label)
        return (a_imgs + new_imgs, a_labels + new_labels)
   
    # rotating 90, 180 and 270 degree
    new_imgs, new_labels = augment(imgs, labels, [imrotate, imrotate, imrotate], [90, 180, 270])

    # flip up-down, left-right
    new_imgs, new_labels = augment(new_imgs, new_labels, [np.flipud, np.fliplr], [None, None])
    
    # shift 1 pixel each direction
    new_imgs, new_labels = augment(new_imgs, new_labels, [shift, shift, shift, shift], 
                                   [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)])  
    return np.array(new_imgs), np.array(new_labels)
