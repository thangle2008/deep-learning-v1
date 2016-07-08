from train_conv import Network
from tools.image_processing import load_image
from configs import alexnet, dinc_sx3_ffc_b32

import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import cPickle
import argparse
from importlib import import_module
import numpy as np

CLASSES = 9
DIM = 140
CROP_DIM = 128
CHANNELS = 3
TRAIN_BATCH_SIZE = 32
IMG_DIR = '../image-data/compressed/bird_full_no_cropped_no_empty_140_rgb.pkl.gz'

# Hyperparameters
ETA = 0.1
LMBDA = 0.0001
EPOCHS = 50

def build_network(model_name, data_size, cudnn=False):
    model = None

    if model_name == 'alexnet':
        model = alexnet.build_model(data_size, CLASSES, cudnn=cudnn, batch_norm=True)
    elif model_name == 'dinc':
        model = dinc_sx3_ffc_b32.build_model(data_size, CLASSES)
    elif model_name == 'googlenet':
        googlenet = import_module('configs.googlenet')
        model = googlenet.build_model(data_size, CLASSES)

    return Network(model['input'], model['output'])
    

def main(args):
    train_data, val_data, test_data, label_dict, img_mean = load_image(args.data, dim=args.dim, mode="RGB",
                                                    zero_center=args.zero_center,
                                                    train_size=args.train_size,
                                                    crop=args.crop)  
    
    # use both val and test as val
    val_data = (np.concatenate((val_data[0], test_data[0]), axis = 0),
                np.concatenate((val_data[1], test_data[1])))

    num_channels = train_data[0][0].shape[0]
    
    data_size = (None, num_channels, CROP_DIM, CROP_DIM)

    def objective(hyperargs):
        net = build_network(args.model, data_size, cudnn=args.dnn)
        lr = args.learning_rate
        lmbda = args.lmbda

        if 'lr' in hyperargs:
            lr = hyperargs['lr']
        if 'lmbda' in hyperargs:
            lmbda = hyperargs['lmbda']
        
        best_val_cost, train_costs, val_costs = net.train(algorithm, train_data, val_data=val_data, test_data=None,
                                        lr=lr, lmbda=lmbda, train_batch_size=TRAIN_BATCH_SIZE, 
                                        val_batch_size=10, epochs = args.epoch, crop_dim=CROP_DIM,
                                        img_mean=img_mean, color_jitter=args.color_jitter)
        return {'loss': best_val_cost, 'status': STATUS_OK}

    space = {}
    if not args.learning_rate:
        print "Optimizing learning rate"
        space['lr'] = hp.uniform('lr', 0, 0.001)
    if not args.lmbda:
        print "Optimizing regularization rate"
        space['lmbda'] = hp.uniform('lambda', 0, 0.1)

    trials = Trials()

    best = fmin(objective, space, algo=tpe.suggest, max_evals=args.num_trials, trials=trials)

    print best
    print hyperopt.space_eval(space, best)
    
    f = open('out.txt', 'w')
    cPickle.dump(trials.trials, f)
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--algorithm', dest='algorithm')
    parser.add_argument('-e', '--epoch', dest='epoch', action="store", type=int, default=300)
    parser.add_argument('-m', '--model', dest='model')
    parser.add_argument('--learning_rate', dest='learning_rate', action="store", type=float)
    parser.add_argument('--lambda', dest='lmbda', action="store", type=float)
    parser.add_argument('--data', dest='data')
    parser.add_argument('--dim', dest='dim', action="store", type=int, default=160)
    parser.add_argument('--train_size', dest='train_size', action="store", type=float)
    parser.add_argument('--zero_center', dest='zero_center', action="store_true")
    parser.add_argument('--dnn', dest='dnn', action="store_true")
    parser.add_argument('--num_trials', dest='num_trials', action="store", type=int, default=50)
    parser.add_argument('--crop', dest='crop', action="store_true")
    parser.add_argument('--color_jitter', dest='color_jitter', action="store_true")

    args = parser.parse_args()
    
    main(args)
