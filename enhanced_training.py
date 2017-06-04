import os

import numpy as np
from os.path import exists
from sys import stdout
import scipy.misc
import utils
from argparse import ArgumentParser
import tensorflow as tf
import transform
from stylize_image import ffwd
from histmatch import hist_match
NETWORK_PATH='./trained'


#tensor flow training
from cs231n.data_utils import load_CIFAR10

def get_CIFAR10_data(source, num_training=49000, num_validation=1000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = source
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    #mean_image = np.mean(X_train, axis=0)
    #X_train -= mean_image
    #X_val -= mean_image
    #X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--source', type=str,
                        dest='content_path', help='source images dir',
                        metavar='CONTENT', 
                        default='cs231n/datasets/cifar-10-batches-py')

    parser.add_argument('--network-path', type=str,
                        dest='network_path',
                        help='path to network (default %(default)s)',
                        metavar='NETWORK_PATH', default=NETWORK_PATH)

    parser.add_argument('--output-path', type=str,
                        dest='output_path',
                        help='path for output',
                        metavar='OUTPUT_PATH', default = '.')

    #python enhanced_training.py --source=

    # parse and get network directory
    options = parser.parse_args()
    network = options.network_path
    source = options.content_path

    #get cifar10
    X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data(source)
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    #initialize output
    N, H,W,C = X_train.shape 
    enhanced = np.zeros((N,H*4, W*4, C))
    #X_train=X_train.reshape((N,C,H,W)).transpose(0,2,3,1).astype("uint8")

    #generate enhanced samples
    for i in range(X_train.shape[0]):
        content_image = X_train[i]
        content_image = np.squeeze(content_image).reshape(H,W,C)
        #print(content_image.shape)
        content_image = np.ndarray.reshape(content_image, (1,) + content_image.shape)
        prediction = ffwd(content_image, network).reshape((H*4, W*4, C))
        enhanced[i] = prediction
        content_image = np.squeeze(content_image).reshape(H,W,C)
        #print(prediction.shape)
        for c in range(3):
            prediction[:, :, c] = hist_match(prediction[:, :, c], content_image[:, :, c])
        scipy.misc.imsave('super.jpg', prediction)
        #img.save('first.png')
        break


