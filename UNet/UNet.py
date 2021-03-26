
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans

import tensorflow as tf

batchsize = 1
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    #return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

def createConvRelu(input, kenelsize, inchannel, outchannel):
    W_conv1 = weight_variable([kenelsize, kenelsize, inchannel, outchannel])
    b_conv1 = bias_variable([outchannel])
    h_conv1 = tf.nn.relu(conv2d(input, W_conv1) + b_conv1)
    return h_conv1


def create2ConvReluWithPool(input, convKernelSize, inchannel, outchannel):
    conv1 = createConvRelu(input, convKernelSize, inchannel, outchannel)
    conv2 = createConvRelu(conv1, convKernelSize, outchannel, outchannel)
    pool3 = max_pool_2x2(conv2)
    return pool3,conv2

def create2ConvReluDropout(input, convKernelSize, inchannel, outchannel, keepprob):
    conv1 = createConvRelu(input, convKernelSize, inchannel, outchannel)
    conv2 = createConvRelu(conv1, convKernelSize, outchannel, outchannel)
    drop3 = tf.nn.dropout(conv2, keepprob)
    return drop3

def create2ConvReluDropoutPool(input, convKernelSize, inchannel, outchannel, keepprob):
    conv1 = createConvRelu(input, convKernelSize, inchannel, outchannel)
    conv2 = createConvRelu(conv1, convKernelSize, outchannel, outchannel)
    drop3 = tf.nn.dropout(conv2, keepprob)
    pool4 = max_pool_2x2(drop3)
    return pool4,conv2

def downSide(input, firstchannels, keep_prob):
    # input w 256 h 256 channel 1
    unet_out1,unet_conv1 = create2ConvReluWithPool(input, 3, 1, firstchannels)
    # hidden conv1 output w 128 h 128 channel 64
    unet_out2, unet_conv2 = create2ConvReluWithPool(unet_out1, 3, firstchannels, firstchannels*2)
    # hidden pool1 output w 64 h 64 channel 128
    unet_out3,unet_conv3 = create2ConvReluWithPool(unet_out2, 3, firstchannels*2, firstchannels*4)
    # hidden drop1 output w 32 h 32 channel 256
    unet_out4,unet_conv4 = create2ConvReluDropoutPool(unet_out3, 3, firstchannels*4, firstchannels*8, keep_prob)
    # hidden drop1 output w 16 h 16 channel 512
    unet_out5 = create2ConvReluDropout(unet_out4, 3, firstchannels*8, firstchannels*16, keep_prob)
    # hidden drop1 output w 8 h 8 channel 1024
    print("downside net struct")
    print("1 conv shape", unet_conv1.get_shape().as_list())
    print("1 out shape", unet_out1.get_shape().as_list())
    print("2 conv shape", unet_conv2.get_shape().as_list())
    print("2 out shape", unet_out2.get_shape().as_list())
    print("3 conv shape", unet_conv3.get_shape().as_list())
    print("3 out shape", unet_out3.get_shape().as_list())
    print("4 conv shape", unet_conv4.get_shape().as_list())
    print("4 out shape", unet_out4.get_shape().as_list())
    print("5 out shape", unet_out5.get_shape().as_list())

    return [unet_conv1,unet_conv2,unet_conv3,unet_conv4,unet_out5]

def upsampleSideLayer(input, convKernelSize, inchannel, outchannel, merge):
    # should equals to downside kernel
    W_conv1 = weight_variable([convKernelSize, convKernelSize, inchannel, inchannel])
    inputShape = input.get_shape().as_list()
    deconv1 = tf.nn.conv2d_transpose(input,W_conv1,output_shape=[inputShape[0], inputShape[1]*2, inputShape[2]*2, inputShape[3] ],strides=[1,2,2,1],padding="SAME")
    #print("1 deconv shape", deconv1.get_shape().as_list())
    conv2 = createConvRelu(deconv1, convKernelSize-1, inchannel, outchannel)
    #print("2 conv shape", conv2.get_shape().as_list())
    concat3 = tf.concat([conv2, merge], 3)
    #print("3 concat shape", concat3.get_shape().as_list())
    conv4 = createConvRelu(concat3, convKernelSize, inchannel, outchannel)
    #print("4 conv shape", conv4.get_shape().as_list())
    conv5 = createConvRelu(conv4, convKernelSize, outchannel, outchannel)
    #print("5 conv shape", conv5.get_shape().as_list())
    return conv5


def createConvSigmod(input, kenelsize, inchannel, outchannel):
    W_conv1 = weight_variable([kenelsize, kenelsize, inchannel, outchannel])
    b_conv1 = bias_variable([outchannel])
    h_conv1 = tf.nn.sigmoid(conv2d(input, W_conv1) + b_conv1)
    return h_conv1

def upsampleSide(inputs, destchannels):
    unet_out1 = upsampleSideLayer(inputs[4], 3, destchannels*16, destchannels*8, inputs[3])
    unet_out2 = upsampleSideLayer(unet_out1, 3, destchannels*8, destchannels*4, inputs[2])
    unet_out3 = upsampleSideLayer(unet_out2, 3, destchannels*4, destchannels*2, inputs[1])
    unet_out4 = upsampleSideLayer(unet_out3, 3, destchannels*2, destchannels, inputs[0])
    unet_out5 = createConvRelu(unet_out4, 3, destchannels, 2)
    unet_out6 = createConvSigmod(unet_out5, 1, 2, 1)
    print("upside net struct")
    print("1 out shape", unet_out1.get_shape().as_list())
    print("2 out shape", unet_out2.get_shape().as_list())
    print("3 out shape", unet_out3.get_shape().as_list())
    print("4 out shape", unet_out4.get_shape().as_list())
    print("5 out shape", unet_out5.get_shape().as_list())
    print("6 out shape", unet_out6.get_shape().as_list())
    return unet_out6

def adjustData(img,mask):
    img = img / 255
    mask = mask /255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return (img,mask)



def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr



def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img


Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255

def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)


def unetMain():
    sess = tf.InteractiveSession()

    level1Channels = 4 # 64
    #y_ = tf.placeholder("float", shape=[None, 256,256,1])
    y_ = tf.placeholder("float", shape=[batchsize, 256,256,1])

    #x_image = tf.placeholder("float", shape=[None, 256,256,1])
    x_image = tf.placeholder("float", shape=[batchsize, 256,256,1])
    
    keep_prob = tf.placeholder("float")
    downSideLayers = downSide(x_image, level1Channels, keep_prob)
    upsideLayer = upsampleSide(downSideLayers, level1Channels)
    bce = tf.keras.losses.BinaryCrossentropy()
    #cross_entropy = -tf.reduce_sum(y_*tf.log(h_fc2))
    cross_entropy = bce(y_, upsideLayer)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    sess.run(tf.global_variables_initializer())
    #sess.run(tf.initialize_all_variables())

    data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
    trainImages,trainMaskImages = geneTrainNpy("data/membrane/train/aug/","data/membrane/train/aug/")

    for i in range(500):
        print("train step %d "%(i+1))
        for startIndex in range(0,len(trainImages)):
            train_step.run(feed_dict={x_image: trainImages[startIndex:startIndex+batchsize], y_: trainMaskImages[startIndex:startIndex+batchsize], keep_prob: 0.5})

    results = upsideLayer.eval(feed_dict={x_image: trainImages[0:1], keep_prob: 1.0})
    saveResult("data/membrane/test",results)

unetMain()

