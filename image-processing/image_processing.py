import os
import re

import numpy as np
import random
import cPickle
import gzip

from scipy.misc import imread, imresize, imrotate
from scipy.ndimage.interpolation import zoom, shift

def load_image(folder, dim=28, expand=False):
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
                image = imread(filepath, mode="L")
                image_resized = imresize(image, (dim, dim))
                images.append(image_resized)
                categories.append(c)
        c += 1
    
    # expanding the data if desired
    if expand:
        images, categories = expand_data_set(images, categories)
 
    # shuffle the data and make training, validation and test sets
    index_shuf = range(len(images)) 
    random.shuffle(index_shuf)
    images_shuf = [images[i] for i in index_shuf]
    categories_shuf = [categories[i] for i in index_shuf]

    n = len(images_shuf)
    training_x = np.asarray(images_shuf[:60*n/100], dtype=np.float32) / 255
    training_y = np.asarray(categories_shuf[:60*n/100])
    validation_x = np.asarray(images_shuf[60*n/100:80*n/100], dtype=np.float32) / 255
    validation_y = np.asarray(categories_shuf[60*n/100:80*n/100])
    test_x = np.asarray(images_shuf[80*n/100:], dtype=np.float32) / 255
    test_y = np.asarray(categories_shuf[80*n/100:])

    training_data = (training_x, training_y)
    validation_data = (validation_x, validation_y)
    test_data = (test_x, test_y)

    return (training_data, validation_data, test_data)

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
    return new_imgs, new_labels
