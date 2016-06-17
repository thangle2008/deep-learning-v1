import numpy as np
import lasagne
from train_conv import load_data, Network
from configs import dinc_sx3_ffc_b32

params = np.load('learned/best_params.npy')
print params.shape

IMG_DIR = '../image-data/compressed/bird_full_no_cropped_no_empty_140_rgb.pkl.gz'

def main():
    model = dinc_sx3_ffc_b32.build_model((None, 3, 128, 128), 9)
    lasagne.layers.set_all_param_values(model['output'], params)
    net = Network(model['input'], model['output'])

    return net
