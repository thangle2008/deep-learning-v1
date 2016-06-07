import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, Pool2DLayer, DropoutLayer 

import matplotlib.pyplot as plt

from train_conv import load_data, Network

EPOCHS = 100
DIM = 28
CLASSES = 10

train_data, val_data, test_data = load_data('./data/bird_image_full_expanded.pkl.gz')
num_train_examples = train_data[0].shape[0]
num_val_examples = val_data[0].shape[0]

train_data = (train_data[0].reshape(num_train_examples, 1, DIM, DIM), train_data[1])
val_data = (val_data[0].reshape(num_val_examples, 1, DIM, DIM), val_data[1])

data_size = (None, 1, DIM, DIM)
output_size = CLASSES

input_var = T.tensor4('input') 

l_in = InputLayer(data_size, input_var=input_var)

conv1 = Conv2DLayer(l_in, 20, 5)
pool1 = Pool2DLayer(conv1, 2)

conv2 = Conv2DLayer(pool1, 40, 5)
pool2 = Pool2DLayer(conv2, 2)

fc1 = DenseLayer(pool2, 1000)
dropout1 = DropoutLayer(fc1, p=0.5)

fc2 = DenseLayer(dropout1, 1000)
dropout2 = DropoutLayer(fc2, p=0.5)

out = DenseLayer(dropout2, output_size, nonlinearity=lasagne.nonlinearities.softmax)

net = Network(input_var, out)
train_costs, val_costs = net.train(train_data, val_data=val_data, lr=0.03, train_batch_size=10, 
                                    val_batch_size=10, epochs = EPOCHS, train_cost_cached=True,
                                    val_cost_cached=True)

it = range(EPOCHS)

plt.plot(it, train_costs, 'r', it, val_costs, 'b')

plt.show()
