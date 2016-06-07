import theano
import theano.tensor as T
import lasagne.layers
from lasagne.regularization import regularize_network_params, l1, l2

import numpy as np

import gzip
import cPickle

def load_data(filepath):
    f = gzip.open(filepath, 'r')
    train_data, val_data, test_data = cPickle.load(f)
    
    def cast_to_32(data):
        return (data[0].astype(np.float32), data[1].astype(np.int32))

    f.close()
    
    return (cast_to_32(train_data), cast_to_32(val_data), cast_to_32(test_data))
 
class Network:
    def __init__(self, input_var, out_layer):
        self.input_var = input_var
        self.out_layer = out_layer

        # defining testing function
        target_var = T.ivector()
        test_prediction = lasagne.layers.get_output(self.out_layer, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)

        test_loss = test_loss.mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

        self.__test_fn = theano.function([self.input_var, target_var], [test_loss, test_acc])
        self.__get_preds = theano.function([self.input_var], test_prediction)
    
    def train(self, train_data, val_data=None, lr=0.1, lmbda=0.0,
              train_batch_size=None, val_batch_size=None, epochs=1):
        # extract data and labels
        train_x, train_y = train_data
        val_x, val_y = val_data
        if not train_batch_size:
            mini_batch_size = train_x.shape[0]
        if not val_batch_size:
            val_batch_size = val_x.shape[0]

        # calculate number of minibatchs        
        num_examples = train_x.shape[0]

        num_train_batches = num_examples / train_batch_size
        num_val_batches = num_examples / val_batch_size
        
        # cost function
        target_var = T.ivector('targets')
        prediction = lasagne.layers.get_output(self.out_layer)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var).mean()
        
        # regularization
        penalty = regularize_network_params(self.out_layer, l2)
        loss += lmbda * penalty

        # update rule
        params = lasagne.layers.get_all_params(self.out_layer, trainable=True)
        updates = lasagne.updates.sgd(loss, params, learning_rate=lr)
        
        # defining training and testing function
        train_fn = theano.function([self.input_var, target_var], loss, updates=updates) 
        
        # training
        for epoch in xrange(epochs):
            for batch in xrange(num_train_batches):
                iteration = num_train_batches * epoch + batch
                batch_x = train_x[batch*train_batch_size:(batch+1)*train_batch_size]
                batch_y = train_y[batch*train_batch_size:(batch+1)*train_batch_size]
                train_fn(batch_x, batch_y)
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))

                if (iteration+1) % num_train_batches == 0:
                    acc = self.accuracy(val_data, mini_batch_size=val_batch_size)
                    print("Current test accuracy is {0}%".format(acc * 100))
         
    def accuracy(self, test_data, mini_batch_size=None):
        test_x, test_y = test_data
        if not mini_batch_size:
            mini_batch_size = test_x.shape[0]
        num_test_batches = test_x.shape[0] / mini_batch_size
        
        return np.mean([self.__test_fn(test_x[i*mini_batch_size:(i+1)*mini_batch_size],
                                       test_y[i*mini_batch_size:(i+1)*mini_batch_size])[1]
                        for i in xrange(num_test_batches)]) 
