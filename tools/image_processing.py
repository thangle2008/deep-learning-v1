import os
import re

import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from skimage.color import rgb2gray

import random
import cPickle
import gzip

from scipy.misc import imread, imresize, imrotate, toimage, fromimage
from scipy.ndimage.interpolation import zoom, shift

import PIL
from PIL.ImageEnhance import Brightness, Contrast, Color 

import multiprocessing

LOAD_SEED = 41
SEED1 = 42
SEED2 = 43

random.seed(LOAD_SEED)

#### Processing functions
def center_crop(img, dim, axis):
    """
    Center crop an image along an axis.
    """
    offset = (img.shape[axis] - dim) / 2
    if axis == 0:
        return img[offset:offset+dim]
    else:
        return img[:, offset:offset+dim]

def img_resize(img, dim):
    """
    Do Alexnet's cropping style.
    """
    dim1 = img.shape[0]
    dim2 = img.shape[1]
    
    if dim1 < dim:
        img = imresize(img, (dim, dim2))
        dim1 = dim
    if dim2 < dim:
        img = imresize(img, (dim1, dim)) 
        dim2 = dim

    if dim1 == dim and dim2 == dim:
        return img
    elif dim1 > dim2:
        scaling_factor = dim2 / dim
        img = imresize(img, ((dim1 / scaling_factor), dim))
        return center_crop(img, dim, axis=0)
    else:
        scaling_factor = dim1 / dim
        img = imresize(img, (dim, (dim2 / scaling_factor)))
        return center_crop(img, dim, axis=1)

def process_image(imgfile, label, dim, mode, crop=False, color_jitter=False):
    img = imread(imgfile, mode=mode)
    if color_jitter:
        img = color_jitter_func(img)
    if crop:
        return img_resize(img, dim), label
    else:
        return imresize(img, (dim, dim)), label

#### Color jittering functions
def adjust(img, alpha, etype):
    if alpha == 0.0:
        return img

    pil_img = toimage(img)

    enhancer = None
    if etype == "brightness":
        enhancer = Brightness(pil_img)
    elif etype == "color":
        enhancer = Color(pil_img)
    elif etype == "contrast":
        enhancer = Contrast(pil_img)

    return fromimage(enhancer.enhance(alpha))

def color_jitter_func(img):
    """
    Randomly adjust brightness, contrast and color of the image.
    """
    etypes = ['brightness', 'contrast', 'color']

    # Random the order of enhancement 
    random.shuffle(etypes)

    # Adjust consecutively
    new_img = np.array(img)
    for e in etypes:
        alpha = random.uniform(0.5, 1.5)
        new_img = adjust(new_img, alpha, e)

    return new_img
    
#### Main load and save functions
def load_image(folder, dim=140, mode="RGB", train_size=1.0, zero_center=False, crop=False, color_jitter=False):
    print "Loading data"
    images = []
    categories = []
    label_dict = {}
    c = 0

    for root, dirnames, filenames in os.walk(folder):  
        dirnames.sort()
        if root == folder:
            continue
        current_dir = root.split('/')[-1]
        label_dict[c] = current_dir

        print current_dir
        for filename in filenames:
            if re.search("\.(jpg|png|jpeg)$", filename):
                filepath = os.path.join(root, filename)
                images.append(filepath)
                categories.append(c)
        c += 1
   
    # loading and preprocessing images 
    pool = multiprocessing.Pool()
    labeled_images = zip(images, categories) 
    results = [pool.apply_async(process_image, (li[0], li[1], dim, mode, crop, color_jitter))
                        for li in labeled_images]
    labeled_images = [r.get() for r in results]

    images, categories = zip(*labeled_images)
    images = np.asarray(images)
    categories = np.array(categories, dtype = np.int32)
    
    # normalize    
    images = images / np.float32(255)

    # swap axes of the images to format (c, w, h)
    images = images.swapaxes(1, 3).swapaxes(2, 3)
 
    if train_size == 1.0:
        mean_activity_train = None
        if zero_center:
            mean_activity_train = np.mean(images, axis = 0)
            images -= mean_activity_train
            np.save('image-mean', mean_activity_train)
        return (images, categories), None, None, label_dict, mean_activity_train
    else: 
        # stratified shuffle and split the data set
        sss = StratifiedShuffleSplit(categories, 1, test_size=1-train_size,
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

        mean_activity_train = None
        if zero_center:
            mean_activity_train = np.mean(train_x, axis = 0)
            train_x -= mean_activity_train
            val_x -= mean_activity_train
            test_x -= mean_activity_train
            np.save('image-mean', mean_activity_train)

        print train_x.shape

        train_data = (train_x, train_y)
        val_data = (val_x, val_y)
        test_data = (test_x, test_y)

        return train_data, val_data, test_data, label_dict, mean_activity_train

def save_image(data, filename="data_set.pkl.gz"):
    print "Saving data"
    f = gzip.open(filename, "w")
    cPickle.dump(data, f)
    f.close() 

