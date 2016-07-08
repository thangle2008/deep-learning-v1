Using convolutional network to classify images

Usage:

1) To train the network, run the file run.py with these options (you should specify all of them):

|Option         | Argument |
|-------        |----------|
|-a             |algorithm to use (adam, momentum, ...)|
|-e             |number of epochs, default is 300|
|-m             |model to use|
|--learning_rate| desired learning rate |
|--lambda       | regularization rate|
|--data         | path to data folder|
|--dim          | resized dimension, default is 160|
|--train_size   | use this portion of data for training, default is 0.6|
|--zero_center  | whether to subtract the mean of training data |
|--dnn          | whether to run the model with cuDNN |
|--crop         | whether to use Alexnet's cropping style at preprocessing step |
|--color_jitter | whether to change each training image's brightness, contrast and saturation |

2) After the training finishes, there will be 2 important files popping up in the same directory: the image-mean.npy and best-params.npy, which store the mean of the training images (if zero_center flag is set to True) and the best learned weights. You can use these files with the script produce_false_images.py to test on other data set.
