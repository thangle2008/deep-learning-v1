import theano
import theano.tensor as T
import lasagne.layers
from lasagne.regularization import regularize_network_params, l1, l2

import numpy as np
import sklearn.utils

import gzip
import cPickle
import random

CROP_SEED = 28

def load_data(filepath):
    print "loading data"
    f = gzip.open(filepath, 'r')
    train_data, val_data, test_data = cPickle.load(f)
    
    def cast_to_32(data):
	data_x = np.asarray(data[0], dtype=np.float32)
	data_y = np.asarray(data[1], dtype=np.int32)
	return (data_x, data_y)

    f.close()
    
    return (cast_to_32(train_data), cast_to_32(val_data), cast_to_32(test_data))

def random_crop(data, dim, new_dim):
    new_data = []
    num_channels = data[0].shape[0]

    offset = dim - new_dim
    for img in data:
        idx = random.randint(0, offset)
        idy = random.randint(0, offset)

        new_img = img[:, idx:idx+new_dim, idy:idy+new_dim]

        if random.randint(0, 1) == 0:
            new_img = np.fliplr(new_img)
        
        new_img = np.reshape(new_img, (num_channels, new_dim, new_dim))
        new_data.append(new_img)
    return np.asarray(new_data)
        
class Network:
    def __init__(self, l_in, l_out):
        self.l_in = l_in
        self.l_out = l_out

        # defining testing function
        target_var = T.ivector()
        test_prediction = lasagne.layers.get_output(self.l_out, deterministic=True)
        test_cost = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)

        test_cost = test_cost.mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

        self.__test_fn = theano.function([self.l_in.input_var, target_var], [test_cost, test_acc])
        self.__get_preds = theano.function([self.l_in.input_var], test_prediction)

        random.seed(CROP_SEED)
    
    def train(self, train_data, val_data=None, lr=0.1, lmbda=0.0,
              train_batch_size=None, val_batch_size=None, epochs=1,
              train_cost_cached=False, val_cost_cached=False,
              crop_dim=None):
        # extract data and labels
        train_x, train_y = train_data
        val_x, val_y = val_data
        if not train_batch_size:
            mini_batch_size = train_x.shape[0]
        if not val_batch_size:
            val_batch_size = val_x.shape[0]

        # calculate number of minibatches        
        num_examples = train_x.shape[0]

        num_train_batches = num_examples / train_batch_size
        num_val_batches = num_examples / val_batch_size
        
        # cost function
        target_var = T.ivector('targets')
        prediction = lasagne.layers.get_output(self.l_out)
        cost_fn = lasagne.objectives.categorical_crossentropy(prediction, target_var).mean()
        
        # regularization
        penalty = regularize_network_params(self.l_out, l2)
        cost_fn += lmbda * penalty

        # update rule
        params = lasagne.layers.get_all_params(self.l_out, trainable=True)

        updates = lasagne.updates.adagrad(cost_fn, params, learning_rate=lr)
        
        # defining training and testing function
        train_fn = theano.function([self.l_in.input_var, target_var], cost_fn, updates=updates) 
        
        # training
        train_costs, val_costs = [], []
        best_val_acc = 0.0

        for epoch in xrange(epochs):
            current_epoch_costs = []
#            train_x, train_y = sklearn.utils.shuffle(train_x, train_y)
            for batch in xrange(num_train_batches):
                iteration = num_train_batches * epoch + batch
                batch_x = train_x[batch*train_batch_size:(batch+1)*train_batch_size]
                batch_y = train_y[batch*train_batch_size:(batch+1)*train_batch_size]
                
                if crop_dim:
                    dim = batch_x.shape[2]
                    batch_x = random_crop(batch_x, dim, crop_dim)
                train_cost = train_fn(batch_x, batch_y)
                current_epoch_costs.append(train_cost)
#                if iteration % 1000 == 0:
#                    print("Training mini-batch number {0}".format(iteration))

                if (iteration+1) % num_train_batches == 0:
                    val_cost, val_acc = self.cost_and_accuracy(val_data, mini_batch_size=val_batch_size,
                                                               crop_dim=crop_dim)
                    average_train_cost = np.mean(current_epoch_costs)
                    print("Epoch {0} validation accuracy is {1}%".format(epoch, val_acc * 100))
                    print("Average training cost is {0}".format(average_train_cost))
                    
                    if best_val_acc < val_acc:
                        best_val_acc = val_acc
                    
                    if train_cost_cached: 
                        train_costs.append(average_train_cost)
                    if val_cost_cached:
                        val_costs.append(val_cost)
        
        print("Best validation accuracy is {0}%".format(best_val_acc * 100))
        
        return train_costs, val_costs
         
    def cost_and_accuracy(self, test_data, mini_batch_size=None, crop_dim=None):
        test_x, test_y = test_data
        
        if not mini_batch_size:
            mini_batch_size = test_x.shape[0]
        num_test_batches = test_x.shape[0] / mini_batch_size
        
        test_accs = []
        
        for i in xrange(num_test_batches):
            batch_x = test_x[i*mini_batch_size:(i+1)*mini_batch_size]
            batch_y = test_y[i*mini_batch_size:(i+1)*mini_batch_size]
            
            if crop_dim:
                dim = batch_x.shape[2]
                batch_x = random_crop(batch_x, dim, crop_dim)

            test_accs.append(self.__test_fn(batch_x, batch_y))
    
        return np.mean(test_accs, axis=0)
