# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from scipy import io
import tensorflow as tf
#from importlib import reload

import Model_TRANSNet
import TFRecordOp
import HddDataOperation as hdd
import fft_tools

#reload(models)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#reload(module)
DataSize = 100
batchSize = 8#8
row = 256
column = 256
channel = 1
num_epoch = 50

Iter = DataSize // batchSize  

low_holder = tf.placeholder(tf.float32,[batchSize,row,column,channel])
label_holder = tf.placeholder(tf.float32,[batchSize,row,column,channel])
mask_holder = tf.placeholder(tf.complex64,[batchSize,row,column,channel])
train_holder = tf.placeholder(tf.bool)
learn_rate = tf.placeholder(tf.float32)

p0 = fft_tools.fft2d(label_holder) * mask_holder
result = Model_TRANSNet.TRANSNet(low_holder, mask_holder, p0, dropout_rate=1.0, train_sign=True, reuse=None)

lossl2 = hdd.lossL2(result,label_holder)

lossim = 1 - tf.reduce_mean(tf.image.ssim(result,label_holder,1.0))
lossl1 = hdd.lossL1(result,label_holder)
loss = lossl1


trainlossl2 = tf.summary.scalar('train_lossl2',lossl2)
trainlossim = tf.summary.scalar('train_lossim',lossim)
trainloss = tf.summary.scalar('train_loss',loss)
trainlossl1 = tf.summary.scalar('train_lossl1',lossl1)

merge_loss = tf.summary.merge([trainlossl2, trainlossim, trainloss, trainlossl1])

optimizer = tf.train.AdamOptimizer(learn_rate)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train = optimizer.minimize(loss)
    
d_vars = tf.trainable_variables()
lambd_vars = [var for var in d_vars if 'lambda' in var.name]
mu_vars = [var for var in d_vars if 'mu' in var.name]
lambda_clip = [p.assign(tf.nn.relu(p)) for p in lambd_vars]
mu_clip = [p.assign(tf.nn.relu(p)) for p in mu_vars]    
    
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
saver = tf.train.Saver(max_to_keep=500)

check_dir = './TRANSNetCode/checkpoints/'    

print("Initialing Network:success \nTraining....")
'''训练'''

writer = tf.summary.FileWriter('./TRANSNetCode/logdir/',sess.graph)
  
filenames = tf.train.match_filenames_once('./TFRecord/train*.tfrecord')
#    
img_batch, label_batch = TFRecordOp.input_pipeline(filenames, batch_size=batchSize,
        num_epochs=None, num_features_input=[row,column,channel],
        num_features_label=[row,column,channel])

if not os.path.exists(check_dir):
    os.makedirs(check_dir)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
else : 
    chkpt_fname = tf.train.latest_checkpoint(check_dir)
    saver.restore(sess, chkpt_fname) 
    sess.run(tf.local_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)


mask = io.loadmat('./GaussianDistribution2DMask_10.mat')['maskRS2']
mask_reshape = np.reshape(mask, [1, row, column, 1])
masktile = np.tile(mask_reshape, [batchSize, 1, 1, 1])

print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

try:
    for epoch in range(0,num_epoch):  
        if epoch < 20:
            curr_lr = 1e-4
        elif epoch < 30:
            curr_lr = 1e-5
        elif epoch < 40:
            curr_lr = 1e-5
        else:
            curr_lr = 1e-6
        for iter in range(Iter):
            samp_low, samp_label = sess.run([img_batch, label_batch])
            mloss,mlossl2,mlossl1,_,mergeloss,_,_ =\
                          sess.run([loss,lossl2,lossl1,train,merge_loss,lambda_clip,mu_clip],
                            feed_dict={low_holder:samp_low,label_holder:samp_label,learn_rate:curr_lr, mask_holder:masktile,train_holder:True})
            if (iter % 100 == 0):
                print("epoch: %3d Iter %7d loss: %10.5f lossl2: %10.5f lossl1: %10.5f" 
                      % (epoch,iter,mloss,mlossl2,mlossl1))
            if (iter % 100 == 0):
                writer.add_summary(mergeloss,epoch*Iter + iter)
        saver.save(sess, os.path.join(check_dir,"TRANS-Net"), global_step = epoch)
    

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    coord.request_stop()
    coord.join(threads)    
writer.close()












