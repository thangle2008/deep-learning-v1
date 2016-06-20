import numpy as np
import matplotlib.pyplot as plt

import lasagne

from train_conv import load_data, Network
from configs import alexnet, dinc_sx3_ffc_b32, googlenet

import argparse

def main(args):
    train_data, val_data, test_data = load_data(args.data)
    val_test_data = (np.concatenate((val_data[0], test_data[0]), axis=0),
                     np.concatenate((val_data[1], test_data[1])))

    params = np.load(args.clf)
        
    model = None
    if args.model == 'alexnet':
        model = alexnet.build_model((None, 3, 128, 128), 9)
    elif args.model == 'dinc':
        model = dinc_sx3_ffc_b32.build_model((None, 3, 128, 128), 9)
    elif args.model == 'googlenet':
        model = googlenet.build_model((None, 3, 128, 128), 9)

    lasagne.layers.set_all_param_values(model['output'], params)
    net = Network(model['input'], model['output'])
    
    imgs, wrong_labels, correct_labels = net.get_wrong_classification(val_test_data, 10, 128)
     
    # output the wrong labels
    text1 = plt.text(10, 10, "Wrong label", color='r', weight='bold')
    text2 = plt.text(20, 20, "Correct label", color='r', weight='bold')

    for i in xrange(len(imgs)):
        img = imgs[i].swapaxes(1, 2).swapaxes(0, 2)
        text1.set_text("Classified as {0}".format(wrong_labels[i]))
        text2.set_text("Should be {0}".format(correct_labels[i]))
        plt.imshow(img)
        plt.savefig("{0}.png".format(i))
          
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data', dest='data')
    parser.add_argument('--classifier', dest='clf')
    parser.add_argument('-m', '--model', dest='model')

    args = parser.parse_args()
    main(args)      
