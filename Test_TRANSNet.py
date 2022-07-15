# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy import io
import tensorflow as tf
#from importlib import reload

import Model_TRANSNet
import TFRecordOp
import fft_tools

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#reload(module)
row = 256
column = 256
channel = 1

low_holder_valid = tf.placeholder(tf.float32,[1,row,column,channel])
label_holder_valid = tf.placeholder(tf.float32,[1,row,column,channel])
mask_holder_valid = tf.placeholder(tf.complex64,[1,row,column,channel])
p0_valid = fft_tools.fft2d(label_holder_valid) * mask_holder_valid
result = Model_TRANSNet.TRANSNet(low_holder_valid, mask_holder_valid, p0_valid, dropout_rate=0., train_sign=False)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
saver = tf.train.Saver()

check_dir = 'path-to-trained-model'    
saver.restore(sess, check_dir)

mask = io.loadmat('./GaussianDistribution2DMask_10.mat')['maskRS2']
mask_reshape = np.reshape(mask, [1, row, column, 1])

loss = 0
for l in range(200):
    print(l)
    fileName = './TestFile/%05d.npy' % l
    validfull = TFRecordOp.scale_to_01(np.float32(np.load(fileName)))
    validlow = TFRecordOp.to_bad_image(validfull, mask)
    valid_input = np.reshape(validlow,[1,256,256,1])
    valid_label = np.reshape(validfull,[1,256,256,1])
    output = sess.run(result, feed_dict={low_holder_valid:valid_input,label_holder_valid:valid_label, mask_holder_valid:mask_reshape})
    loss = loss + np.square(output[0,:,:,0]-validfull).mean()
print(loss/200)













