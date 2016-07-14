from configs import dinc_sx3_ffc_b32
import numpy as np
import lasagne
import theano
from tools.image_processing import load_image
from train_conv import center_crop
import matplotlib.pyplot as plt

def main(idx):
    lasagne.random.set_rng(np.random.RandomState(1234))
    a, b, c, d, e = load_image('../image-data/test-images', crop=True, dim=160)
    net = dinc_sx3_ffc_b32.build_model((None, 3, 128, 128), 9, batch_norm=False)
    params = np.load('../learned/dinc-44.npy')
    mean = np.load('../means/image-mean-44.npy')
    
    img = a[0][idx]
    plt.imshow(np.array(img).transpose(1, 2, 0))
    plt.savefig('original.png')
    img -= mean
    img = center_crop(np.array([img]), 128)
    mean_crop = center_crop(np.array([mean]), 128)[0] 
    y = lasagne.layers.get_output(net['pool3'], deterministic=True)
    f = theano.function([net['input'].input_var], y)
    
    pooled = f(img)
    pooled_shape = pooled[0][0].shape
    #flat_pooled = pooled.ravel()
    #sorted_idx = np.argsort(flat_pooled)
    #max_9 = flat_pooled[sorted_idx[-100:]]
    #flat_pooled.fill(0)
    #flat_pooled[sorted_idx[-100:]] = max_9
    #pooled = flat_pooled.reshape(pooled.shape)
    #print np.count_nonzero(pooled)

    deconv_net3 = dinc_sx3_ffc_b32.build_separate_deconv(net['pool3'], net['conv3'])
    deconv_net2 = dinc_sx3_ffc_b32.build_separate_deconv(net['pool2'], net['conv2'], deconv_net3['deconv'])
    deconv_net1 = dinc_sx3_ffc_b32.build_separate_deconv(net['pool1'], net['conv1'], deconv_net2['deconv'])    
    y = lasagne.layers.get_output(deconv_net1['deconv'], deterministic=True)
    f = theano.function([net['input'].input_var, deconv_net3['input'].input_var], y)

    num_feature = pooled.shape[1]
    for i in range(num_feature):
        p = np.array(pooled)
        p[:, 0:i] = np.zeros(pooled_shape)
        p[:, i+1:] = np.zeros(pooled_shape)
        res = f(img, p) 
        res[res == 0] = 1
        plt.imshow(res[0].transpose(1, 2, 0))
        plt.savefig('{0}.png'.format(i))
