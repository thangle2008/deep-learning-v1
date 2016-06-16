from train_conv import load_data, Network
from configs import alex_net, dinc_sx3_ffc_b32

import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import cPickle

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

def build_network():
    data_size = (None, CHANNELS, CROP_DIM, CROP_DIM)

#    model = dinc_sx3_ffc_b32.build_model(data_size, CLASSES)
    model = alex_net.build_model_revised(data_size, CLASSES)

    return Network(model['input'], model['output'])
    

def main():
    train_data, val_data, test_data = load_data(IMG_DIR)
    
    def objective(lr):
        net = build_network()
        lmbda = LMBDA
        num_epoch = EPOCHS

        best_val_cost, _, _ = net.train('adam', train_data, val_data=val_data, test_data=test_data,
                                        lr=lr, lmbda=lmbda, train_batch_size=TRAIN_BATCH_SIZE, 
                                        val_batch_size=10, epochs = num_epoch, train_cost_cached=False,
                                        val_cost_cached=False, crop_dim=CROP_DIM)
        return {'loss': best_val_cost, 'status': STATUS_OK}

    space = hp.uniform('lr', 0, 0.001)
    trials = Trials()

    best = fmin(objective, space, algo=tpe.suggest, max_evals=100, trials=trials)

    print best
    print hyperopt.space_eval(space, best)
    
    f = open('out.txt', 'w')
    cPickle.dump(trials.trials, f)
    f.close()

if __name__ == '__main__':
    main()
