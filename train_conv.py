import theano
import theano.tensor as T
import lasagne.layers
from lasagne.regularization import regularize_network_params, l1, l2

import gzip
import cPickle

def load_data(filepath):
    f = gzip.open(filepath, 'r')
    data = cPickle.load(f)
    f.close()
    
    return data     
    
class Network:
    def __init__(self, input_var, out_layer):
        self.input_var = input_var
        self.out_layer = out_layer
    
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

        test_prediction = lasagne.layers.get_output(self.out_layer, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)

        test_loss = test_loss.mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

        val_fn = theano.function([self.input_var, target_var], [test_loss, test_acc])
        get_preds = theano.function([self.input_var], test_prediction)
        
        # training
        for epoch in xrange(epochs):
            for batch in xrange(num_train_batches):
                iteration = num_train_batches * epoch + batch
                x_batch = train_x[batch*train_batch_size:(batch+1)*train_batch_size]
                y_batch = train_y[batch*train_batch_size:(batch+1)*train_batch_size]
                train_fn(x_batch, y_batch)

                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))

                if (iteration+1) % num_train_batches == 0:
                    loss, acc = val_fn(test_x, test_y) 
                    print("Current test accuracy is {0}%".format(acc * 100))
