Using convolutional network to classify images

Usage:

1) First, you should go to the image_processing directory and execute the script run.py to convert the image folder into the desired format:

python run.py --img_dir [path to img folder] --dim [resized dimension, default is 140]

You will see a compressed file popping up.

2) Go back to the parent directory and run the script run.py. The options you have to specify are:

|Option     | Argument |
|-------    |----------|
|-a         |algorithm to use (adam, momentum, ...)|
|-e         |number of epochs (300, ..)|
|-m         |model to use (alexnet or dinc)|
|--learning_rate| desired learning rate |
|--lambda       | regularization rate|
|--data         | path to compressed data (ex: sth/data.pkl.gz)|

