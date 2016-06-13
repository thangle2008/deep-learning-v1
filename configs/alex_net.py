import theano.tensor as T
import lasagne

def build_model(data_size, num_classes):
    net = {}
    input_var = T.tensor4('input') 

    net['input'] = lasagne.layers.InputLayer(
      shape=data_size,
      )

    net['conv1'] = lasagne.layers.Conv2DLayer(
      net['input'],
      num_filters=32,
      filter_size=8,
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform(gain='relu'),
      ) 
    net['norm1'] = lasagne.layers.LocalResponseNormalization2DLayer(net['conv1'])
    net['pool1'] = lasagne.layers.MaxPool2DLayer(net['norm1'], 
      pool_size=(2, 2),
      stride=2,
      ) #out 67x67x32

    net['conv2'] = lasagne.layers.Conv2DLayer(
      net['pool1'],
      num_filters=64,
      filter_size=7,
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform(gain='relu'),
      b=lasagne.init.Constant(1.0),
      ) 
    net['norm2'] = lasagne.layers.LocalResponseNormalization2DLayer(net['conv2'])
    net['pool2'] = lasagne.layers.MaxPool2DLayer(net['norm2'], 
      pool_size=(2, 2),
      stride=2,
      ) #out 31x31x64

    net['conv3-1'] = lasagne.layers.Conv2DLayer(
      net['pool2'],
      num_filters=128,
      filter_size=3,
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform(gain='relu'),
      ) #out 29x29x128
    net['conv3-2'] = lasagne.layers.Conv2DLayer(
      net['conv3-1'],
      num_filters=128,
      filter_size=3,
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform(gain='relu'),
      b=lasagne.init.Constant(1.0),
      ) #out 27x27x128

    net['conv4'] = lasagne.layers.Conv2DLayer(
      net['conv3-2'],
      num_filters=64,
      filter_size=3,
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform(gain='relu'),
      b=lasagne.init.Constant(1.0),
      )
    net['pool4'] = lasagne.layers.MaxPool2DLayer(net['conv4'], 
      pool_size=(2, 2),
      stride=2,
      ) #out 13x13x64


    net['fc1'] = lasagne.layers.DenseLayer(
      net['pool4'],
      num_units=512,
      W=lasagne.init.GlorotUniform(gain="relu"),
      b=lasagne.init.Constant(1.0),
      )

    net['dropout1'] = lasagne.layers.DropoutLayer(net['fc1'], p=0.5)


    net['fc2'] = lasagne.layers.DenseLayer(
      net['dropout1'],
      num_units=512,
      W=lasagne.init.GlorotUniform(gain="relu"),
      b=lasagne.init.Constant(1.0),
      )
    net['dropout2'] = lasagne.layers.DropoutLayer(net['fc2'], p=0.5)

    # - applies the softmax after computing the final layer units
    net['output'] = lasagne.layers.DenseLayer(
      net['dropout2'],
      #l_pool3,
      num_units=num_classes,
      nonlinearity=lasagne.nonlinearities.softmax,
      #W=lasagne.init.GlorotUniform(),
      )
    return net

def build_model_revised(data_size, num_classes):
    net = {}
    input_var = T.tensor4('input') 

    net['input'] = lasagne.layers.InputLayer(
      shape=data_size,
      )

    net['conv1'] = lasagne.layers.Conv2DLayer(
      net['input'],
      num_filters=32,
      filter_size=8,
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform(gain='relu'),
      ) 
    net['norm1'] = lasagne.layers.LocalResponseNormalization2DLayer(net['conv1'])
    net['pool1'] = lasagne.layers.MaxPool2DLayer(net['norm1'], 
      pool_size=(2, 2),
      stride=2,
      ) #out 32x60x60

    net['conv2'] = lasagne.layers.Conv2DLayer(
      net['pool1'],
      num_filters=64,
      filter_size=8,
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform(gain='relu'),
      ) 
    net['norm2'] = lasagne.layers.LocalResponseNormalization2DLayer(net['conv2'])
    net['pool2'] = lasagne.layers.MaxPool2DLayer(net['norm2'], 
      pool_size=(2, 2),
      stride=2,
      ) #out 64x26x26

    net['conv3-1'] = lasagne.layers.Conv2DLayer(
      net['pool2'],
      num_filters=128,
      filter_size=3,
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform(gain='relu'),
      ) #out 128x24x24
    net['conv3-2'] = lasagne.layers.Conv2DLayer(
      net['conv3-1'],
      num_filters=128,
      filter_size=3,
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform(gain='relu'),
      ) #out 128x22x22

    net['conv4'] = lasagne.layers.Conv2DLayer(
      net['conv3-2'],
      num_filters=64,
      filter_size=2,
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform(gain='relu'),
      )
    net['pool4'] = lasagne.layers.MaxPool2DLayer(net['conv4'], 
      pool_size=(2, 2),
      stride=2,
      ) #out 64x10x10


    net['fc1'] = lasagne.layers.DenseLayer(
      net['pool4'],
      num_units=512,
      W=lasagne.init.GlorotUniform(gain="relu"),
      )

    net['dropout1'] = lasagne.layers.DropoutLayer(net['fc1'], p=0.5)


    net['fc2'] = lasagne.layers.DenseLayer(
      net['dropout1'],
      num_units=512,
      W=lasagne.init.GlorotUniform(gain="relu"),
      )
    net['dropout2'] = lasagne.layers.DropoutLayer(net['fc2'], p=0.5)

    # - applies the softmax after computing the final layer units
    net['output'] = lasagne.layers.DenseLayer(
      net['dropout2'],
      num_units=num_classes,
      nonlinearity=lasagne.nonlinearities.softmax,
      #W=lasagne.init.GlorotUniform(),
      )
    return net

