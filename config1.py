import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, Pool2DLayer, DropoutLayer 

from train_conv import load_data, Network

train_data, val_data, test_data = load_data('./data/mnist.pkl.gz')
train_data = (train_data[0].reshape(50000, 1, 28, 28), train_data[1])
val_data = (val_data[0].reshape(10000, 1, 28, 28), val_data[1])

data_size = (None, 1, 28, 28)
output_size = 10

input_var = T.tensor4('input') 

l_in = InputLayer(data_size, input_var=input_var)

conv1 = Conv2DLayer(l_in, 20, 5)
pool1 = Pool2DLayer(conv1, 2)

conv2 = Conv2DLayer(pool1, 40, 5)
pool2 = Pool2DLayer(conv2, 2)

fc1 = DenseLayer(pool2, 100)
dropout = DropoutLayer(fc1, p=0.5)
out = DenseLayer(dropout, output_size, nonlinearity=lasagne.nonlinearities.softmax)

net = Network(input_var, out)
net.train(train_data, val_data=val_data, lr=0.03, train_batch_size=10, val_batch_size=100, 
        epochs = 3)
