import tensorflow as tf
import os 
import numpy as np
from PIL import Image
import glob
import skimage.io as io
import skimage.transform as trans

def get_files():
    datas = []
    label_datas = []

    srcImg = Image.open("1.bmp")
    datas.append( np.array(srcImg).reshape(640, 480, 1)/255 )
    destImg = Image.open("1_result.bmp")
    label_datas.append( np.array(destImg).reshape(640, 480, 1)/255 )
    srcImg = Image.open("2.bmp")
    datas.append( np.array(srcImg).reshape(640, 480, 1)/255 )
    destImg = Image.open("2_result.bmp")
    label_datas.append( np.array(destImg).reshape(640, 480, 1)/255 )
    srcImg = Image.open("3.bmp")
    datas.append( np.array(srcImg).reshape(640, 480, 1)/255 )
    destImg = Image.open("3_result.bmp")
    label_datas.append( np.array(destImg).reshape(640, 480, 1)/255 )

    srcImg = Image.open("4.bmp")
    datas.append( np.array(srcImg).reshape(640, 480, 1)/255 )
    destImg = Image.open("4_result.bmp")
    label_datas.append( np.array(destImg).reshape(640, 480, 1)/255 )
    srcImg = Image.open("5.bmp")
    datas.append( np.array(srcImg).reshape(640, 480, 1)/255 )
    destImg = Image.open("5_result.bmp")
    label_datas.append( np.array(destImg).reshape(640, 480, 1)/255 )
    srcImg = Image.open("6.bmp")
    datas.append( np.array(srcImg).reshape(640, 480, 1)/255 )
    destImg = Image.open("6_result.bmp")
    label_datas.append( np.array(destImg).reshape(640, 480, 1)/255 )
    
    return datas, label_datas


def saveResult(save_path,npyfile, idx):
    for i,item in enumerate(npyfile):
        img = item[:,:,0]
        img = img.reshape(480,640, 1)
        img = img*255
        io.imsave(os.path.join(save_path,"%d_%d_predict.bmp"%(idx, i)),img)


sess = tf.InteractiveSession()


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def encodLayer(input , kernelsize , inchannels, outchannels, keepprob ):
    W_conv = weight_variable([kernelsize, kernelsize, inchannels, outchannels]) # 3x3 1 in 32 out
    b_conv = bias_variable([outchannels])
    conv = tf.nn.relu(conv2d(input, W_conv) + b_conv)
    drop = tf.nn.dropout(conv, keepprob)
    pool = tf.nn.max_pool(drop, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return pool

def decodLayer(input , kernelsize, inchannels, outchannels, keepprob):    
    W_conv = weight_variable([kernelsize, kernelsize, inchannels, outchannels]) # 3x3 32 in 32 out
    b_conv = bias_variable([outchannels])
    conv = tf.nn.relu(conv2d(input, W_conv) + b_conv)
    drop = tf.nn.dropout(conv, keepprob)
    inputShape = conv.get_shape().as_list()
    W_dconv = weight_variable([kernelsize, kernelsize, outchannels, outchannels]) # 3x3 32 in 32 out
    deconv = tf.nn.conv2d_transpose(drop,W_dconv,output_shape=[inputShape[0], inputShape[1]*2, inputShape[2]*2, inputShape[3] ],strides=[1,2,2,1],padding="SAME")
    return deconv

channel = 32
epoch = 20000

keep_prob = tf.placeholder("float")
x = tf.placeholder(tf.float32, shape=[1, 640, 480, 1])
y_ = tf.placeholder(tf.float32, shape=[1, 640, 480, 1])

# encode 
encodeLayer = encodLayer(x , 3 , 1, channel, keep_prob ) # 320 x 240 
encodeLayer = encodLayer(encodeLayer , 3, channel, channel, keep_prob ) # 160 x 120 
encodeLayer = encodLayer(encodeLayer , 3, channel, channel, keep_prob ) # 80 x 60 
encodeLayer = encodLayer(encodeLayer , 3, channel, channel, keep_prob ) # 40 x 30 
encodeLayer = encodLayer(encodeLayer , 3, channel, channel, keep_prob ) # 20 x 15 

# decode
decodeLayer = encodeLayer
decodeLayer = decodLayer(decodeLayer , 3, channel, channel, keep_prob)
decodeLayer = decodLayer(decodeLayer , 3, channel, channel, keep_prob)
decodeLayer = decodLayer(decodeLayer , 3, channel, channel, keep_prob)
decodeLayer = decodLayer(decodeLayer , 3, channel, channel, keep_prob)
decodeLayer = decodLayer(decodeLayer , 3, channel, channel, keep_prob)

W_decodeLayer1 = weight_variable([3, 3, channel, 1]) # 3x3 32 in 1 out
b_decodeLayer1 = bias_variable([1])
decodeLayer1 = tf.nn.sigmoid(conv2d(decodeLayer, W_decodeLayer1) + b_decodeLayer1)

# loss
#learning_rate = 0.01
#momentum = 0.9
#accuracy = tf.reduce_mean(tf.square(decodeLayer1 - y_))
#train_step = tf.train.MomentumOptimizer(
#    learning_rate = learning_rate, 
#    momentum = momentum, 
#    use_nesterov = True
#).minimize(accuracy)

bce = tf.keras.losses.BinaryCrossentropy()
accuracy = bce(y_, decodeLayer1)
train_step = tf.train.AdamOptimizer(1e-4).minimize(accuracy)

# sess.run(tf.initialize_all_variables())
sess.run(tf.global_variables_initializer())

train, result = get_files()

#print(train[0])
#exit()
i = 0
#for i in range(20000):
for i in range(epoch):
#while True:
    for startIndex in range(len(train)):
        #train_step.run({x:train[startIndex:startIndex+1], y_: result[startIndex:startIndex+1], keep_prob:1.0})
        train_step.run({x:train[startIndex:startIndex+1], y_: train[startIndex:startIndex+1], keep_prob: 0.7})
    i+=1
    if i%50 == 0:
        print( "step %d"%(i) )
    if (i+1)%200 == 0:
        results = decodeLayer1.eval(feed_dict={x:train[0:1], keep_prob:1.0})
        saveResult("out",results, i+1)
        #if train_accuracy<0.0008:
        #    break
    #    print( "step %d"%(i) )

#print( "test accuracy %g"%accuracy.eval(feed_dict={x:train, y_: result, keep_prob:1.0}) )


results = decodeLayer1.eval(feed_dict={x:train[0:1], keep_prob:1.0})
#print(results.as_list())
saveResult("out",results, i)

