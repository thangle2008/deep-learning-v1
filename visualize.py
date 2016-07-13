from configs import alexnet
from train_conv import center_crop
import numpy as np
import theano
import lasagne
import matplotlib.pyplot as plt
from tools.image_processing import load_image

net = alexnet.build_model((None, 3, 128, 128), 9, batch_norm=True)
params = np.load('../learned/alexnet-42.npy')
mean = np.load('../means/image-mean-42.npy')

a, b, c, d, e = load_image('../image-data/test-images', dim=160, crop=True)

img = a[0][1]
img -= mean
img = np.array([img])
img = center_crop(img, 128)

y1 = lasagne.layers.get_output(net['conv1'], deterministic=True)
y2 = lasagne.layers.get_output(net['conv2'], deterministic=True)
y3 = lasagne.layers.get_output(net['conv4'], deterministic=True)

f = theano.function([net['input'].input_var], [y1, y2, y3])

conv1, conv2, conv4 = f(img)
conv1, conv2, conv4 = conv1[0], conv2[0], conv4[0]

num = 0
for i in conv1:
    plt.imshow(i, cmap='gray')
    plt.savefig('./outputs/conv1/{0}.png'.format(num))
    num += 1

num = 0
for i in conv2:
    plt.imshow(i, cmap='gray')
    plt.savefig('./outputs/conv2/{0}.png'.format(num))
    num += 1

num = 0
for i in conv4:
    plt.imshow(i, cmap='gray')
    plt.savefig('./outputs/conv4/{0}.png'.format(num))
    num += 1
