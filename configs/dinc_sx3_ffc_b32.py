import theano.tensor as T
import lasagne

def build_model(data_size, num_classes):
    net = {}
    input_var = T.tensor4('input') 

    net['input'] = lasagne.layers.InputLayer(
      shape=data_size,
      )

    net['pad1'] = lasagne.layers.PadLayer(
      net['input'],
      width=1,#padding width
      )

    net['conv1'] = lasagne.layers.Conv2DLayer(
      net['pad1'],
      num_filters=32,
      filter_size=3,
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform(gain='relu'),
      )

    net['pool1'] = lasagne.layers.MaxPool2DLayer(net['conv1'], 
      pool_size=(2, 2),
      stride=2,
      )


    net['pad2'] = lasagne.layers.PadLayer(
      net['pool1'],
      width=1,#padding width
      )

    net['conv2'] = lasagne.layers.Conv2DLayer(
      net['pad2'],
      num_filters=64,
      filter_size=3,
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform(gain='relu'),
      )

    net['pool2'] = lasagne.layers.MaxPool2DLayer(net['conv2'], 
      pool_size=(2, 2),
      stride=2,
      )


    net['pad3'] = lasagne.layers.PadLayer(
      net['pool2'],
      width=1,#padding width
      )

    net['conv3'] = lasagne.layers.Conv2DLayer(
      net['pad3'],
      num_filters=128,
      filter_size=3,
      nonlinearity=lasagne.nonlinearities.rectify,
      W=lasagne.init.GlorotUniform(gain='relu'),
      )


    net['pool3'] = lasagne.layers.MaxPool2DLayer(net['conv3'], 
      pool_size=(2, 2),
      stride=2,
      )


    net['fc1'] = lasagne.layers.DenseLayer(
      net['pool3'],
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
      #l_pool3,
      num_units=num_classes,
      nonlinearity=lasagne.nonlinearities.softmax,
      #W=lasagne.init.GlorotUniform(),
      )

    return net
