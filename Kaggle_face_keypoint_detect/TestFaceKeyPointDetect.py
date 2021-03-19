from datetime import datetime
import os
import sys

from matplotlib import pyplot
import numpy as np

from pandas import DataFrame
from pandas.io.parsers import read_csv

#from sklearn.utils import shuffle

FTRAIN = 'training.csv'
FTEST = 'test.csv'
FLOOKUP = 'IdLookupTable.csv'

def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        #X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

def load2d(test=False, cols=None):
    X, y = load(test=test, cols=cols)
    X = X.reshape(-1, 96, 96, 1)
    return X, y



###############################################################################################
#
#  tensorflow functions
#

import tensorflow as tf
sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    #return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    #return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

def useCnnLayer():
    learning_rate = 0.01
    momentum = 0.9
    #x = tf.placeholder("float", shape=[None, 9216])
    y_ = tf.placeholder("float", shape=[None, 30])

    #x_image = tf.reshape(x, [-1,96,96,1])
    x_image = tf.placeholder("float", shape=[None, 96,96,1])

    keep_prob = tf.placeholder("float")
    # input w 96 h 96 channel 1
    W_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # hidden conv1 output w 94 h 94 channel 32
    h_pool1 = max_pool_2x2(h_conv1)
    # hidden pool1 output w 47 h 47 channel 32
    h_pool_drop1 = tf.nn.dropout(h_pool1, keep_prob)
    # hidden drop1 output w 47 h 47 channel 32

    W_conv2 = weight_variable([2, 2, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool_drop1, W_conv2) + b_conv2)
    # hidden conv2 output w 46 h 46 channel 64
    h_pool2 = max_pool_2x2(h_conv2)
    # hidden pool2 output w 23 h 23 channel 64
    h_pool_drop2 = tf.nn.dropout(h_pool2, keep_prob)
    # hidden drop2 output w 23 h 23 channel 64

    W_conv3 = weight_variable([2, 2, 64, 128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_pool_drop2, W_conv3) + b_conv3)
    # hidden conv3 output w 22 h 22 channel 128
    h_pool3 = max_pool_2x2(h_conv3)
    # hidden pool3 output w 11 h 11 channel 128
    h_pool_drop3 = tf.nn.dropout(h_pool3, keep_prob)
    # hidden drop3 output w 11 h 11 channel 128
    
    W_fc1 = weight_variable([11 * 11 * 128, 100])
    b_fc1 = bias_variable([100])
    h_pool3_flat = tf.reshape(h_pool_drop3, [-1, 11 * 11 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([100, 30])
    b_fc2 = bias_variable([30])
    h_fc2=tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    '''
    W_conv4 = weight_variable([2, 2, 128, 256])
    b_conv4 = bias_variable([256])
    h_conv4 = tf.nn.relu(conv2d(h_pool_drop3, W_conv4) + b_conv4)
    # hidden conv4 output w 10 h 10 channel 256
    h_pool4 = max_pool_2x2(h_conv4)
    # hidden pool4 output w 5 h 5 channel 256
    h_pool_drop4 = tf.nn.dropout(h_pool4, keep_prob)
    # hidden drop4 output w 5 h 5 channel 256

    W_fc1 = weight_variable([5 * 5 * 256, 300])
    b_fc1 = bias_variable([300])
    h_pool4_flat = tf.reshape(h_pool_drop4, [-1, 5 * 5 * 256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([300, 30])
    b_fc2 = bias_variable([30])
    h_fc2=tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    '''

    #cross_entropy = -tf.reduce_sum(y_*tf.log(h_fc2))
    #train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    cross_entropy = tf.reduce_mean(tf.square(h_fc2 - y_))
    train_step = tf.train.MomentumOptimizer(
        learning_rate = learning_rate, 
        momentum = momentum, 
        use_nesterov = True
    ).minimize(cross_entropy)
    #sess.run(tf.global_variables_initializer())
    sess.run(tf.initialize_all_variables())

    X, result = load2d()

    for i in range(1001):
        print(i)
        for startIndex in range(0,len(X), 50):
            train_step.run(feed_dict={x_image: X[startIndex:startIndex+50], y_: result[startIndex:startIndex+50], keep_prob: 1.0})

    print(result[0:2])
    print("----")
    print( h_fc2.eval(feed_dict={x_image: X[0:2], keep_prob: 1.0}) )
    #print(W_conv1.eval())
    #print(b_conv2.eval())





# use mutiline to test it
def useMultiLayer():
    import tensorflow as tf
    sess = tf.InteractiveSession()

    learning_rate = 0.01
    momentum = 0.9

    y_ = tf.placeholder("float", shape=[None, 30])
    x = tf.placeholder("float", shape=[None, 9216])

    W_fc1 = weight_variable([9216, 100] )
    b_fc1 = bias_variable([100])
    h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

    W_fc2 = weight_variable([100, 30])
    b_fc2 = bias_variable([30] )
    h_fc2=tf.matmul(h_fc1, W_fc2) + b_fc2

    #cross_entropy = -tf.reduce_sum(y_*tf.log(h_fc2))
    cross_entropy = tf.reduce_mean(tf.square(h_fc2 - y_))
    #train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    train_step = tf.train.MomentumOptimizer(
            learning_rate = learning_rate, 
            momentum = momentum, 
            use_nesterov = True
        ).minimize(cross_entropy)

    sess.run(tf.global_variables_initializer())
    #sess.run(tf.initialize_all_variables())

    X, result = load()

    for i in range(500):
        for startIndex in range(0,len(X), 50):
            #print("train %d %d "%(startIndex, startIndex+50))
            train_step.run(feed_dict={x: X[startIndex:startIndex+50], y_: result[startIndex:startIndex+50]})


    print(result[0:2])
    print("----")
    print( h_fc2.eval(feed_dict={x: X[0:2]}) )



useCnnLayer()
