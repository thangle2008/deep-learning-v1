from train_conv import load_data, Network
from configs import alex_net

import matplotlib.pyplot as plt
import numpy as np

EPOCHS = 300
CLASSES = 9
DIM = 140
CROP_DIM = 128
CHANNELS = 3
TRAIN_BATCH_SIZE = 32

data_size = (None, CHANNELS, CROP_DIM, CROP_DIM)

train_data, val_data, test_data = load_data('../image-data/compressed/bird_full_no_cropped_no_empty_140_rgb.pkl.gz')
num_train_examples = train_data[0].shape[0]
num_val_examples = val_data[0].shape[0]
num_test_examples = test_data[0].shape[0]

train_data = (train_data[0].reshape(num_train_examples, CHANNELS, DIM, DIM), train_data[1])
val_data = (val_data[0].reshape(num_val_examples, CHANNELS, DIM, DIM), val_data[1])
test_data = (test_data[0].reshape(num_test_examples, CHANNELS, DIM, DIM), test_data[1])

# use all val and test for validation
#val_data =(np.concatenate((val_data[0], test_data[0]), axis = 0), np.concatenate((val_data[1], test_data[1])))

# build the model
model = alex_net.build_model(data_size, CLASSES)
net = Network(model['input'], model['output'])

train_costs, val_costs = net.train(train_data, val_data=val_data, test_data=test_data,
                                    lr=0.009, lmbda=0.0005, train_batch_size=TRAIN_BATCH_SIZE, 
                                    val_batch_size=10, epochs = EPOCHS, train_cost_cached=True,
                                    val_cost_cached=True, crop_dim=CROP_DIM)

it = range(EPOCHS)

plt.plot(it, train_costs, 'r', it, val_costs, 'b')

plt.savefig('./experiments/alex.png')
plt.show()
