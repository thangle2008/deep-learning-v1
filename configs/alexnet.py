import theano.tensor as T
import lasagne
from lasagne.layers import normalization

from importlib import import_module

def build_model(data_size, num_classes, batch_norm=True, cudnn=False):
    net = {}
    input_var = T.tensor4('input') 

    Conv2DLayer = lasagne.layers.Conv2DLayer
    Pool2DLayer = lasagne.layers.MaxPool2DLayer

    if cudnn:
        print "Running with cuDNN"
        dnn_module = import_module('lasagne.layers.dnn')
        Conv2DLayer = dnn_module.Conv2DDNNLayer
        Pool2DLayer = dnn_module.MaxPool2DDNNLayer

    net['input'] = lasagne.layers.InputLayer(
      shape=data_size,
      )

    net['conv1'] = Conv2DLayer(
      net['input'],
      num_filters=32,
      filter_size=4,
      flip_filters=False,
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform(gain='relu'),
      ) 
    if batch_norm:
        net['conv1'] = normalization.batch_norm(net['conv1'])
    net['norm1'] = lasagne.layers.LocalResponseNormalization2DLayer(net['conv1'])
    net['pool1'] = Pool2DLayer(net['norm1'], 
      pool_size=(2, 2),
      stride=2,
      ) #out 32x62x62

    net['conv2'] = Conv2DLayer(
      net['pool1'],
      num_filters=64,
      pad=1,
      filter_size=4,
      flip_filters=False,
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform(gain='relu'),
      ) 
    if batch_norm:
        net['conv2'] = normalization.batch_norm(net['conv2'])
    net['norm2'] = lasagne.layers.LocalResponseNormalization2DLayer(net['conv2'])
    net['pool2'] = Pool2DLayer(net['norm2'], 
      pool_size=(2, 2),
      stride=2,
      ) #out 64x30x30

    net['conv3-1'] = Conv2DLayer(
      net['pool2'],
      num_filters=128,
      filter_size=3,
      flip_filters=False,
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform(gain='relu'),
      ) #out 128x28x28
    if batch_norm:
        net['conv3-1'] = normalization.batch_norm(net['conv3-1'])
    net['conv3-2'] = Conv2DLayer(
      net['conv3-1'],
      num_filters=128,
      filter_size=3,
      flip_filters=False,
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform(gain='relu'),
      ) #out 128x26x26
    if batch_norm:
        net['conv3-2'] = normalization.batch_norm(net['conv3-2'])

    net['conv4'] = Conv2DLayer(
      net['conv3-2'],
      num_filters=64,
      pad=1,
      filter_size=4,
      flip_filters=False,
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform(gain='relu'),
      )
    if batch_norm:
        net['conv4'] = normalization.batch_norm(net['conv4'])
    net['pool4'] = Pool2DLayer(net['conv4'], 
      pool_size=(2, 2),
      stride=2,
      ) #out 64x12x12


    net['fc1'] = lasagne.layers.DenseLayer(
      net['pool4'],
      num_units=512,
      W=lasagne.init.GlorotUniform(gain="relu"),
      )
    if batch_norm:
        net['fc1'] = normalization.batch_norm(net['fc1'])
    net['dropout1'] = lasagne.layers.DropoutLayer(net['fc1'], p=0.5)


    net['fc2'] = lasagne.layers.DenseLayer(
      net['dropout1'],
      num_units=512,
      W=lasagne.init.GlorotUniform(gain="relu"),
      )
    if batch_norm:
        net['fc2'] = normalization.batch_norm(net['fc2'])
    net['dropout2'] = lasagne.layers.DropoutLayer(net['fc2'], p=0.5)

    # - applies the softmax after computing the final layer units
    net['output'] = lasagne.layers.DenseLayer(
      net['dropout2'],
      num_units=num_classes,
      nonlinearity=lasagne.nonlinearities.softmax,
      #W=lasagne.init.GlorotUniform(),
      )
    if batch_norm:
        net['output'] = normalization.batch_norm(net['output'])
    return net
