import theano
import theano.tensor as T
import lasagne.layers
from lasagne.regularization import regularize_network_params, l1, l2

import numpy as np
import sklearn.utils

import gzip
import cPickle
import random

from tools.image_processing import color_jitter_func

CROP_SEED = 28
SHUFFLE_SEED = 29

# Augmentation helper
def random_crop(img, new_dim):
    """
    Randomly crop and horizontally flip (with prob 0.5) an image.
    """
    dim = img.shape[1]
    offset = dim - new_dim
    
    idx = random.randint(0, offset)
    idy = random.randint(0, offset)

    new_img = img[:, idx:idx+new_dim, idy:idy+new_dim]

    if random.randint(0, 1) == 0:
        new_img = new_img[:, :, ::-1]

    return new_img

def center_crop(data, new_dim):
    """
    Center crop a set of images.
    """
    dim = data.shape[2]
    offset = (dim - new_dim) / 2
    return data[:, :, offset:offset+new_dim, offset:offset+new_dim]
    
def get_corner_crops(data, new_dim, horizontal_flip=False):
    """
    Return an augmented dataset with images generated by cropping
    each original image, along with its mirror, at 4 corners.
    """
    res = []
    dim = data.shape[2]
    res.append(data[:, :, 0:new_dim, 0:new_dim])
    res.append(data[:, :, dim-new_dim:, 0:new_dim])
    res.append(data[:, :, 0:new_dim, dim-new_dim:])
    res.append(data[:, :, dim-new_dim:, dim-new_dim:])

    if horizontal_flip:
        new_data = []
        for d in res:
            new_data.append(d[:, :, :, ::-1])
        res = res + new_data
    
    return res
    
def augment(data, crop_dim, img_mean=None, color_jitter=False):
    """
    Augment a dataset by random cropping and color jittering (optional).
    """
    new_data = []
    for img in data:
        new_img = np.array(img)
        if color_jitter:
            if img_mean is not None:
                new_img += img_mean
            new_img = new_img.transpose(1, 2, 0)
            new_img = color_jitter_func(new_img)
            new_img = new_img.transpose(2, 0, 1)
            new_img = new_img / np.float32(255)
            if img_mean is not None:
                new_img -= img_mean
        new_img = random_crop(new_img, crop_dim)
    
        new_data.append(new_img)
    return np.array(new_data)

# Main network class
class Network:
    def __init__(self, l_in, l_out):
        self.l_in = l_in
        self.l_out = l_out

        # defining testing function
        target_var = T.ivector()

        test_prediction = lasagne.layers.get_output(self.l_out, deterministic=True)
        self.__get_preds = theano.function([self.l_in.input_var], test_prediction)

        # get cost and acc from predictions
        pred_tensor = T.fmatrix()
        test_cost_pred = lasagne.objectives.categorical_crossentropy(pred_tensor, target_var)
        test_cost_pred = test_cost_pred.mean()
        test_acc_pred = T.mean(T.eq(T.argmax(pred_tensor, axis=1), target_var), dtype=theano.config.floatX)
        
        self.__pred_test_fn = theano.function([pred_tensor, target_var], [test_cost_pred, test_acc_pred])
        random.seed(CROP_SEED)
    
    def train(self, algorithm, train_data, val_data=None, test_data=None, lr=0.1, lmbda=0.0,
              train_batch_size=None, val_batch_size=None, epochs=1,
              train_cost_cached=False, val_cost_cached=False,
              crop_dim=128, color_jitter=False, img_mean=None):
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
        deterministic_prediction = lasagne.layers.get_output(self.l_out, deterministic=True)

        cost_fn = lasagne.objectives.categorical_crossentropy(prediction, target_var).mean()
        deterministic_cost_fn = lasagne.objectives.categorical_crossentropy(deterministic_prediction, 
                                        target_var).mean()
        
        # regularization
        penalty = regularize_network_params(self.l_out, l2)
        cost_fn += lmbda * penalty

        # update rule
        params = lasagne.layers.get_all_params(self.l_out, trainable=True)
        
        updates = None
        if algorithm == 'adagrad':
            updates = lasagne.updates.adagrad(cost_fn, params, learning_rate=lr)
        elif algorithm == 'momentum':
            updates = lasagne.updates.momentum(cost_fn, params, learning_rate=lr, momentum = 0.9)
        elif algorithm == 'adadelta':
            updates = lasagne.updates.adadelta(cost_fn, params, learning_rate=lr)
        elif algorithm == 'adam':
            updates = lasagne.updates.adam(cost_fn, params, learning_rate=lr)
        
        # defining training function
        train_fn = theano.function([self.l_in.input_var, target_var], [cost_fn, deterministic_cost_fn], 
                                     updates=updates) 
        
        # training
        train_costs, val_costs = [], []
        best_val_acc = 0.0
        best_val_cost = float('inf')
        best_params = None

        print("Training network with algorithm={0}, lr={1}, lmbda={2} in {3} epochs".format(algorithm, lr, lmbda, epochs))

        for epoch in xrange(epochs):
            epoch_costs = []
            # shuffle the training data at each epoch
            train_x, train_y = sklearn.utils.shuffle(train_x, train_y, random_state=SHUFFLE_SEED)
            for batch in xrange(num_train_batches):
                iteration = num_train_batches * epoch + batch
                batch_x = train_x[batch*train_batch_size:(batch+1)*train_batch_size]
                batch_y = train_y[batch*train_batch_size:(batch+1)*train_batch_size]
               
                # training data augmenting
                batch_x = augment(batch_x, crop_dim, img_mean=img_mean, color_jitter=color_jitter) 

                _, train_cost = train_fn(batch_x, batch_y)
                epoch_costs.append(train_cost)

                if (iteration+1) % num_train_batches == 0:
                    if val_data:
                        val_cost, val_acc = self.cost_and_accuracy(val_data, 
                                    mini_batch_size=val_batch_size, crop_dim=crop_dim)
                        print("Epoch {0} validation accuracy is {1}%".format(epoch, val_acc * 100))
                        if best_val_acc < val_acc:
                            best_val_acc = val_acc
                            best_params = lasagne.layers.get_all_param_values(self.l_out) 
                            print("This is the best validation accuracy")
                            if test_data:
                                _, test_acc = self.cost_and_accuracy(test_data,
                                    mini_batch_size=val_batch_size, crop_dim=crop_dim, augment_test=False)
                                print("The corresponding test accuracy: {0}%".format(test_acc * 100))

                        best_val_cost = min(best_val_cost, val_cost)
                        average_train_cost = np.mean(epoch_costs)

                        print("Current training cost is: {0}".format(average_train_cost))

                        if val_cost_cached:
                            val_costs.append(val_cost)
                        if train_cost_cached:
                            train_costs.append(average_train_cost)
        
        print("Best validation accuracy is {0}%".format(best_val_acc * 100))
        
        # save the best params
        np.save("best_params", best_params)
 
        return best_val_cost, train_costs, val_costs
        
    def get_batch_predictions(self, test_x, crop_dim=128, augment_test=False):
        """
        Get predictions for each image in a batch.
        """
        cropped = center_crop(test_x, crop_dim)
        
        if augment_test:
            additional_tests = get_corner_crops(test_x, crop_dim, horizontal_flip=True)
            additional_tests.append(cropped)
            
            predictions = []
            for tx in additional_tests:
                predictions.append(self.__get_preds(tx))
            return np.mean(predictions, axis = 0)
        else:
            return self.__get_preds(cropped)

    def cost_and_accuracy(self, test_data, mini_batch_size=10, crop_dim=128, augment_test=False):
        """
        Return the cost and accuracy on this test set.
        """
        test_x, test_y = test_data
        
        if not mini_batch_size:
            mini_batch_size = test_x.shape[0]
        num_test_batches = test_x.shape[0] / mini_batch_size
        
        test_accs = []
        
        for i in xrange(num_test_batches):
            batch_x = test_x[i*mini_batch_size:(i+1)*mini_batch_size]
            batch_y = test_y[i*mini_batch_size:(i+1)*mini_batch_size]
           
            predictions = self.get_batch_predictions(batch_x, crop_dim, augment_test)
            test_accs.append(self.__pred_test_fn(predictions, batch_y))
    
        return np.mean(test_accs, axis=0)

    def get_wrong_classification(self, test_data, mini_batch_size=10, crop_dim=None, augment_test=False):
        """
        Return the set of incorrectly labeled images, 
        """
        test_x, test_y = test_data

        if not mini_batch_size:
            mini_batch_size = test_x.shape[0]
        num_test_batches = test_x.shape[0] / mini_batch_size
        
        res = None
        incorrect_labels = None
        correct_labels = None

        for i in xrange(num_test_batches):
            batch_x = test_x[i*mini_batch_size:(i+1)*mini_batch_size]
            batch_y = test_y[i*mini_batch_size:(i+1)*mini_batch_size]
            
            non_cropped = batch_x
 
            if crop_dim:
                batch_x = center_crop(batch_x, crop_dim)
            
            predictions = np.argmax(self.get_batch_predictions(batch_x, crop_dim, augment_test), axis=1)
            diff_idx = np.where(predictions != batch_y)[0]
 
            if res is not None:
                res = np.concatenate((res, non_cropped[diff_idx]))
                incorrect_labels = np.concatenate((incorrect_labels, predictions[diff_idx]))
                correct_labels = np.concatenate((correct_labels, batch_y[diff_idx]))
            else:
                res = non_cropped[diff_idx]
                incorrect_labels = predictions[diff_idx]
                correct_labels = batch_y[diff_idx]

        return (res, incorrect_labels, correct_labels)
