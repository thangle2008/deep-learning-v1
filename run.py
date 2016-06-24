from train_conv import Network
from configs import alexnet, dinc_sx3_ffc_b32
from tools.image_processing import load_image

import matplotlib.pyplot as plt
import numpy as np

import argparse
import cPickle
import gzip

CLASSES = 9
DIM = 140
CROP_DIM = 128
TRAIN_BATCH_SIZE = 32
EPOCHS = 300

def main(args, optimize=False):
    algorithm = 'adagrad'
    num_epoch = EPOCHS
    model = 'alexnet'
    lr, lmbda = 0.009, 0.0001
    train_size = 0.6

    if args.algorithm:
        algorithm = args.algorithm
    if args.model:
        model = args.model 
    if args.learning_rate:
        lr = args.learning_rate
    if args.lmbda:
        lmbda = args.lmbda
    if args.epoch:
        num_epoch = args.epoch
    if args.train_size:
        train_size = args.train_size

    train_data, val_data, test_data, label_dict = load_image(args.data, dim=args.dim, mode="RGB",
                                                    normalize=args.normalize, zero_center=args.zero_center,
                                                    train_size=train_size)  
                                               
    num_channels = train_data[0][0].shape[0]
    
    data_size = (None, num_channels, CROP_DIM, CROP_DIM)

    # use both val and test as val
    val_data = (np.concatenate((val_data[0], test_data[0]), axis = 0),
                np.concatenate((val_data[1], test_data[1])))

    # build the model
    if model == 'alexnet':
        model = alexnet.build_model_revised(data_size, CLASSES, cudnn=args.dnn)
    elif model == 'dinc':
        model = dinc_sx3_ffc_b32.build_model(data_size, CLASSES)
    elif model == 'googlenet':
        googlenet = import_module('configs.googlenet')
        model = googlenet.build_model(data_size, CLASSES)

    net = Network(model['input'], model['output'])

    best_val_cost, train_costs, val_costs = net.train(algorithm, train_data, val_data=val_data, test_data=test_data,
                                        lr=lr, lmbda=lmbda, train_batch_size=TRAIN_BATCH_SIZE, 
                                        val_batch_size=10, epochs = num_epoch, train_cost_cached=True,
                                        val_cost_cached=True, crop_dim=CROP_DIM)

    it = range(EPOCHS)

    plt.plot(it, train_costs, 'r', it, val_costs, 'b')

    plt.savefig('./alex.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--algorithm', dest='algorithm')
    parser.add_argument('-e', '--epoch', dest='epoch', action="store", type=int)
    parser.add_argument('-m', '--model', dest='model')
    parser.add_argument('--learning_rate', dest='learning_rate', action="store", type=float)
    parser.add_argument('--lambda', dest='lmbda', action="store", type=float)
    parser.add_argument('--data', dest='data')
    parser.add_argument('--dim', dest='dim', action="store", type=int)
    parser.add_argument('--train_size', dest='train_size', action="store", type=float)
    parser.add_argument('--zero_center', dest='zero_center', action="store_true")
    parser.add_argument('--normalize', dest='normalize', action="store_true")
    parser.add_argument('--dnn', dest='dnn', action="store_true")

    args = parser.parse_args()

    main(args)
