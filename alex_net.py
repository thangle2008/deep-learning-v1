import theano.tensor as T
import lasagne

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
#val_data =(np.concatenate((val_data[0], test_data[0]), axis = 0), np.concatenate((val_data[1], test_data[1])))
data_size = (None, CHANNELS, CROP_DIM, CROP_DIM)

print val_data[0].shape

input_var = T.tensor4('input') 

l_in = lasagne.layers.InputLayer(
  shape=data_size,
  )

l_conv1 = lasagne.layers.Conv2DLayer(
  l_in,
  num_filters=32,
  filter_size=8,
  nonlinearity=lasagne.nonlinearities.rectify,
  W=lasagne.init.GlorotUniform(gain='relu'),
  ) 
l_norm1 = lasagne.layers.LocalResponseNormalization2DLayer(l_conv1)
l_pool1 = lasagne.layers.MaxPool2DLayer(l_norm1, 
  pool_size=(2, 2),
  stride=2,
  ) #out 67x67x32

l_conv2 = lasagne.layers.Conv2DLayer(
  l_pool1,
  num_filters=64,
  filter_size=7,
  nonlinearity=lasagne.nonlinearities.rectify,
  W=lasagne.init.GlorotUniform(gain='relu'),
  ) 
l_norm2 = lasagne.layers.LocalResponseNormalization2DLayer(l_conv2)
l_pool2 = lasagne.layers.MaxPool2DLayer(l_norm2, 
  pool_size=(2, 2),
  stride=2,
  ) #out 31x31x64

l_conv3_1 = lasagne.layers.Conv2DLayer(
  l_pool2,
  num_filters=128,
  filter_size=3,
  nonlinearity=lasagne.nonlinearities.rectify,
  W=lasagne.init.GlorotUniform(gain='relu'),
  ) #out 29x29x128
l_conv3_2 = lasagne.layers.Conv2DLayer(
  l_conv3_1,
  num_filters=128,
  filter_size=3,
  nonlinearity=lasagne.nonlinearities.rectify,
  W=lasagne.init.GlorotUniform(gain='relu'),
  ) #out 27x27x128

l_conv4 = lasagne.layers.Conv2DLayer(
  l_conv3_2,
  num_filters=64,
  filter_size=3,
  nonlinearity=lasagne.nonlinearities.rectify,
  W=lasagne.init.GlorotUniform(gain='relu'),
  )
l_pool4 = lasagne.layers.MaxPool2DLayer(l_conv4, 
  pool_size=(2, 2),
  stride=2,
  ) #out 13x13x64


l_fc1 = lasagne.layers.DenseLayer(
  l_pool4,
  num_units=512,
  W=lasagne.init.GlorotUniform(gain="relu"),
  )

l_fc1_dropout = lasagne.layers.DropoutLayer(l_fc1, p=0.5)


l_fc2 = lasagne.layers.DenseLayer(
  l_fc1_dropout,
  num_units=512,
  W=lasagne.init.GlorotUniform(gain="relu"),
  )
l_fc2_dropout = lasagne.layers.DropoutLayer(l_fc2, p=0.5)

# - applies the softmax after computing the final layer units
l_out = lasagne.layers.DenseLayer(
  l_fc2_dropout,
  #l_pool3,
  num_units=CLASSES,
  nonlinearity=lasagne.nonlinearities.softmax,
  #W=lasagne.init.GlorotUniform(),
  )

net = Network(l_in, l_out)
train_costs, val_costs = net.train(train_data, val_data=val_data, test_data=test_data,
                                    lr=0.009, lmbda=0.0005, train_batch_size=TRAIN_BATCH_SIZE, 
                                    val_batch_size=10, epochs = EPOCHS, train_cost_cached=True,
                                    val_cost_cached=True, crop_dim=CROP_DIM)

it = range(EPOCHS)

plt.plot(it, train_costs, 'r', it, val_costs, 'b')

plt.savefig('./experiments/alex.png')
plt.show()
