from train_conv import load_data, Network
from configs import alex_net, dinc_sx3_ffc_b32

import matplotlib.pyplot as plt
import numpy as np

import argparse
import cPickle
import gzip

CLASSES = 9
DIM = 140
CROP_DIM = 128
CHANNELS = 3
TRAIN_BATCH_SIZE = 32
IMG_DIR = '../image-data/compressed/bird_full_no_cropped_no_empty_140_rgb.pkl.gz'
EPOCHS = 300

def main(args, optimize=False):
    algorithm = 'adagrad'
    num_epoch = EPOCHS
    model = 'alexnet'
    lr, lmbda = 0.009, 0.0001

    if args.algorithm:
        algorithm = args.algorithm
    if args.epoch:
        num_epoch = args.epoch
    if args.model:
        model = args.model 
    if args.learning_rate:
        lr = args.learning_rate
    if args.lmbda:
        lmbda = args.lmbda

    data_size = (None, CHANNELS, CROP_DIM, CROP_DIM)

    train_data, val_data, test_data = load_data(IMG_DIR)

    # use both val and test as val
    val_data = (np.concatenate((val_data[0], test_data[0]), axis = 0),
                np.concatenate((val_data[1], test_data[1])))

    # build the model
    if model == 'alexnet':
        model = alex_net.build_model_revised(data_size, CLASSES)
    elif model == 'dinc':
        model = dinc_sx3_ffc_b32.build_model(data_size, CLASSES)

    net = Network(model['input'], model['output'])

    best_val_cost, train_costs, val_costs = net.train(algorithm, train_data, val_data=val_data, test_data=test_data,
                                        lr=lr, lmbda=lmbda, train_batch_size=TRAIN_BATCH_SIZE, 
                                        val_batch_size=10, epochs = num_epoch, train_cost_cached=True,
                                        val_cost_cached=True, crop_dim=CROP_DIM)

    it = range(EPOCHS)

    plt.plot(it, train_costs, 'r', it, val_costs, 'b')

    plt.savefig('./experiments/alex.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--algorithm', dest='algorithm')
    parser.add_argument('-e', '--epoch', dest='epoch', action="store", type=int)
    parser.add_argument('-m', '--model', dest='model')
    parser.add_argument('--learning_rate', dest='learning_rate', action="store", type=float)
    parser.add_argument('--lambda', dest='lmbda', action="store", type=float)
    
    args = parser.parse_args()

    main(args)
