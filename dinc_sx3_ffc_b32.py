import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, Pool2DLayer, DropoutLayer 

import matplotlib.pyplot as plt
import numpy as np

from train_conv import load_data, Network

EPOCHS = 100
CLASSES = 9
DIM = 140
CROP_DIM = 128
CHANNELS = 3
TRAIN_BATCH_SIZE = 32

train_data, val_data, test_data = load_data('../image-data/compressed/bird_full_expanded_no_cropped_no_empty_140_rgb.pkl.gz')
num_train_examples = train_data[0].shape[0]
num_val_examples = val_data[0].shape[0]
num_test_examples = test_data[0].shape[0]

train_data = (train_data[0].reshape(num_train_examples, CHANNELS, DIM, DIM), train_data[1])
val_data = (val_data[0].reshape(num_val_examples, CHANNELS, DIM, DIM), val_data[1])
test_data = (test_data[0].reshape(num_test_examples, CHANNELS, DIM, DIM), test_data[1])

# use all val and test for validation
val_data =(np.concatenate((val_data[0], test_data[0]), axis = 0), np.concatenate((val_data[1], test_data[1])))
data_size = (None, CHANNELS, CROP_DIM, CROP_DIM)

print val_data[0].shape

input_var = T.tensor4('input') 

l_in = lasagne.layers.InputLayer(
  shape=data_size,
  )

l_pad1 = lasagne.layers.PadLayer(
  l_in,
  width=1,#padding width
  )

l_conv1 = lasagne.layers.Conv2DLayer(
  l_pad1,
  num_filters=32,
  filter_size=3,
  nonlinearity=lasagne.nonlinearities.rectify,
  W=lasagne.init.GlorotUniform(gain='relu'),
  )

l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, 
  pool_size=(2, 2),
  stride=2,
  )


l_pad2 = lasagne.layers.PadLayer(
  l_pool1,
  width=1,#padding width
  )

l_conv2 = lasagne.layers.Conv2DLayer(
  l_pad2,
  num_filters=64,
  filter_size=3,
  nonlinearity=lasagne.nonlinearities.rectify,
  W=lasagne.init.GlorotUniform(gain='relu'),
  )

l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, 
  pool_size=(2, 2),
  stride=2,
  )



l_pad3 = lasagne.layers.PadLayer(
  l_pool2,
  width=1,#padding width
  )

l_conv3 = lasagne.layers.Conv2DLayer(
  l_pad3,
  num_filters=128,
  filter_size=3,
  nonlinearity=lasagne.nonlinearities.rectify,
  W=lasagne.init.GlorotUniform(gain='relu'),
  )


l_pool3 = lasagne.layers.MaxPool2DLayer(l_conv3, 
  pool_size=(2, 2),
  stride=2,
  )


l_hidden1 = lasagne.layers.DenseLayer(
  l_pool3,
  num_units=512,
  W=lasagne.init.GlorotUniform(gain="relu"),
  )

l_hidden1_dropout = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)


l_hidden2 = lasagne.layers.DenseLayer(
  l_hidden1_dropout,
  num_units=512,
  W=lasagne.init.GlorotUniform(gain="relu"),
  )

l_hidden2_dropout = lasagne.layers.DropoutLayer(l_hidden2, p=0.5)

# - applies the softmax after computing the final layer units
l_out = lasagne.layers.DenseLayer(
  l_hidden2_dropout,
  #l_pool3,
  num_units=CLASSES,
  nonlinearity=lasagne.nonlinearities.softmax,
  #W=lasagne.init.GlorotUniform(),
  )

net = Network(l_in, l_out)
train_costs, val_costs = net.train(train_data, val_data=val_data, lr=0.009, lmbda=0.0001, train_batch_size=TRAIN_BATCH_SIZE, 
                                    val_batch_size=10, epochs = EPOCHS, train_cost_cached=True,
                                    val_cost_cached=True, crop_dim=CROP_DIM)

it = range(EPOCHS)

plt.plot(it, train_costs, 'r', it, val_costs, 'b')

plt.savefig('./experiments/config1.png')
plt.show()
