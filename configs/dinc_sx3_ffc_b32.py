import theano.tensor as T
import lasagne
from lasagne.layers import normalization
from lasagne.layers import NonlinearityLayer, InverseLayer
from lasagne.layers import TransposedConv2DLayer, ExpressionLayer
import numpy as np

def _set_zeros(activations):
    data_shape = activations[0][0].shape

    res = T.set_subtensor(activations[:, 1:], T.zeros(data_shape))
    return res

def build_deconv(name, conv, pool_layer=None):
    net = {}
    top = net[name+'/express'] = ExpressionLayer(pool_layer, _set_zeros)
    if pool_layer:
        top = net[name+'/pool'] = InverseLayer(top, pool_layer)
    net[name+'/nonlin'] = NonlinearityLayer(top)
    net[name+'/conv'] = TransposedConv2DLayer(net[name+'/nonlin'], conv.input_shape[1],
            conv.filter_size, stride=conv.stride, crop=conv.pad,
            W=conv.W, flip_filters=not conv.flip_filters) 
    return net
     
def build_separate_deconv(pool_layer, conv):
    net = {}
    data_size = lasagne.layers.get_output_shape(pool_layer)
    net['input'] = lasagne.layers.InputLayer(shape=data_size)
    net['unpool'] = InverseLayer(net['input'], pool_layer)
    net['nonlin'] = NonlinearityLayer(net['unpool'])
    net['deconv'] = TransposedConv2DLayer(net['nonlin'], conv.input_shape[1],
            conv.filter_size, stride=conv.stride, crop=conv.pad,
            W=conv.W, flip_filters=not conv.flip_filters) 
    return net

def build_model(data_size, num_classes, batch_norm=True, deconv=False):
    net = {}
    input_var = T.tensor4('input') 

    net['input'] = lasagne.layers.InputLayer(
      shape=data_size,
      )

    net['conv1'] = lasagne.layers.Conv2DLayer(
      net['input'],
      num_filters=32,
      filter_size=3,
      pad=1,
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform(gain='relu'),
      )
    if batch_norm:
        net['conv1'] = normalization.batch_norm(net['conv1'])
    net['pool1'] = lasagne.layers.MaxPool2DLayer(net['conv1'], 
      pool_size=(2, 2),
      stride=2,
      )
    if deconv:
        net.update(build_deconv('deconv1', net['conv1'], net['pool1']))

    net['conv2'] = lasagne.layers.Conv2DLayer(
      net['pool1'],
      num_filters=64,
      filter_size=3,
      pad=1,
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform(gain='relu'),
      )
    if batch_norm:
        net['conv2'] = normalization.batch_norm(net['conv2'])
    net['pool2'] = lasagne.layers.MaxPool2DLayer(net['conv2'], 
      pool_size=(2, 2),
      stride=2,
      )

    net['conv3'] = lasagne.layers.Conv2DLayer(
      net['pool2'],
      num_filters=128,
      filter_size=3,
      pad=1,
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform(gain='relu'),
      )
    if batch_norm:
        net['conv3'] = normalization.batch_norm(net['conv3'])
    net['pool3'] = lasagne.layers.MaxPool2DLayer(net['conv3'], 
      pool_size=(2, 2),
      stride=2,
      )

    net['fc1'] = lasagne.layers.DenseLayer(
      net['pool3'],
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
      )
    if batch_norm:
        net['output'] = normalization.batch_norm(net['output'])

    return net
