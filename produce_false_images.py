import numpy as np
import matplotlib.pyplot as plt
from importlib import import_module

import lasagne

from train_conv import Network
from tools.image_processing import load_image

from configs import alexnet, dinc_sx3_ffc_b32

import argparse

CROP_DIM = 128

def main(args):
    data, _, _, label_dict, _ = load_image(args.data, dim=args.dim, crop=args.crop, color_jitter=args.color_jitter)

    img_mean = None
    if args.img_mean:
        img_mean = np.load(args.img_mean)
        data = (data[0]-img_mean, data[1])

    print data[0].shape

    params = np.load(args.clf)

    num_channels = data[0][0].shape[0]
        
    data_size = (None, num_channels, CROP_DIM, CROP_DIM)
    model = None
    if args.model == 'alexnet':
        model = alexnet.build_model(data_size, 9, cudnn=args.dnn, batch_norm=True)
    elif args.model == 'dinc':
        model = dinc_sx3_ffc_b32.build_model(data_size, 9)
    elif args.model == 'googlenet':
        googlenet = import_module('configs.googlenet')
        model = googlenet.build_model(data_size, 9)

    lasagne.layers.set_all_param_values(model['output'], params)
    net = Network(model['input'], model['output'])
   
    print net.cost_and_accuracy(data, 10, CROP_DIM, augment_test=args.augment_test) 

    if args.output_png:
        imgs, wrong_labels, correct_labels = net.get_wrong_classification(data, 10, CROP_DIM,
                                                augment_test=args.augment_test)
         
        # output the wrong labels
        text1 = plt.text(10, 10, "Wrong label", color='r', weight='bold')
        text2 = plt.text(10, 20, "Correct label", color='r', weight='bold')

        for i in xrange(len(imgs)):
            if img_mean is not None:
                img = imgs[i] + img_mean
            img = img.swapaxes(1, 2).swapaxes(0, 2)
            text1.set_text("Classified as {0}".format(label_dict[wrong_labels[i]]))
            text2.set_text("Should be {0}".format(label_dict[correct_labels[i]]))
            plt.imshow(img)
            plt.savefig("{0}.png".format(i))
          
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data', dest='data')
    parser.add_argument('--classifier', dest='clf')
    parser.add_argument('-m', '--model', dest='model')
    parser.add_argument('--output_png', dest='output_png', action='store_true')
    parser.add_argument('--dim', dest='dim', action='store', type=int, default=160)
    parser.add_argument('--dnn', dest='dnn', action="store_true")
    parser.add_argument('--crop', dest='crop', action="store_true")
    parser.add_argument('--color_jitter', dest='color_jitter', action="store_true")
    parser.add_argument('--img_mean', dest='img_mean')
    parser.add_argument('--augment_test', dest='augment_test', action="store_true")
    
    args = parser.parse_args()
    main(args)      
