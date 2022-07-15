# -*- coding: utf-8 -*-

import tensorflow as tf
import fft_tools
import Transformer

def batch_norm(input_,scope='BN',bn_train=True):

    return tf.contrib.layers.batch_norm(input_,scale=True,epsilon=1e-8,
                                        is_training=bn_train,scope=scope)

    
def conv2d(input, filters, kernel_size, name, strides = (1,1), paddings = 'same',dilation_rate=(1, 1)):
    return tf.layers.conv2d(input, filters=filters, kernel_size=kernel_size, strides=strides,padding=paddings, 
                            dilation_rate=(1, 1),kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                            bias_initializer=tf.constant_initializer(0.01),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                            name=name)

def conv2d_transpose(input,filters,kernel_size,name,paddings='same',strides=[2,2]):
    return tf.layers.conv2d_transpose(input,filters,kernel_size,strides=strides,padding=paddings,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                            bias_initializer=tf.constant_initializer(0.01),name=name)

def lrelu(inputs):
    return tf.nn.relu(inputs)

def uk_block(inputs, train_sign=True):
    basicFilterNumScale = 4
    kernelSize = [3, 3]
    with tf.name_scope('uk_block'):
        with tf.variable_scope('uk_block'):
            
            out1_conv1 = conv2d(inputs,basicFilterNumScale,kernelSize,'conv1')
            out1_bn1 = batch_norm(out1_conv1,'bn1',bn_train=train_sign)
            out1_lrelu1 = lrelu(out1_bn1)
            
            out1_conv2 = conv2d(out1_lrelu1,basicFilterNumScale,kernelSize,'conv2')
            out1_bn2 = batch_norm(out1_conv2,'bn2',bn_train=train_sign)
            out1_lrelu2 = lrelu(out1_bn2)
            
            maxpool_1 = tf.nn.max_pool(out1_lrelu2,[1,2,2,1],[1,2,2,1],'VALID')
            out1_conv3 = conv2d(maxpool_1,2*basicFilterNumScale,kernelSize,'conv3')
            out1_bn3 = batch_norm(out1_conv3,'bn3',bn_train=train_sign)
            out1_lrelu3 = lrelu(out1_bn3)
            out1_conv4 = conv2d(out1_lrelu3,2*basicFilterNumScale,kernelSize,'conv4')
            out1_bn4 = batch_norm(out1_conv4,'bn4',bn_train=train_sign)
            out1_lrelu4 = lrelu(out1_bn4)
            
            maxpool_2 = tf.nn.max_pool(out1_lrelu4,[1,2,2,1],[1,2,2,1],'VALID')
            out1_conv5 = conv2d(maxpool_2,4*basicFilterNumScale,kernelSize,'conv5')
            out1_bn5 = batch_norm(out1_conv5,'bn5',bn_train=train_sign)
            out1_lrelu5 = lrelu(out1_bn5)
            out1_conv6 = conv2d(out1_lrelu5,4*basicFilterNumScale,kernelSize,'conv6')
            out1_bn6 = batch_norm(out1_conv6,'bn6',bn_train=train_sign)
            out1_lrelu6 = lrelu(out1_bn6)
            
#            out1_lrelu6 = Transformer.TransformerMoudle(maxpool_2, dropout_rate, train_sign, 'uk')
            
            upsamp1 = conv2d_transpose(out1_lrelu6,2*basicFilterNumScale,kernelSize,'up1')
            concat1 = tf.concat([out1_lrelu4,upsamp1],3)
            
            out1_conv7 = conv2d(concat1,2*basicFilterNumScale,kernelSize,'conv7')
            out1_bn7 = batch_norm(out1_conv7,'bn7',bn_train=train_sign)
            out1_lrelu7 = lrelu(out1_bn7)
            out1_conv8 = conv2d(out1_lrelu7,2*basicFilterNumScale,kernelSize,'conv8')
            out1_bn8 = batch_norm(out1_conv8,'bn8',bn_train=train_sign)
            out1_lrelu8 = lrelu(out1_bn8)
            
            upsamp2 = conv2d_transpose(out1_lrelu8,basicFilterNumScale,kernelSize,'up2')
            concat2 = tf.concat([out1_lrelu2,upsamp2],3)
            
            out1_conv9 = conv2d(concat2,basicFilterNumScale,kernelSize,'conv9')
            out1_bn9 = batch_norm(out1_conv9,'bn9',bn_train=train_sign)
            out1_lrelu9 = lrelu(out1_bn9)
            out1_conv10 = conv2d(out1_lrelu9,basicFilterNumScale,kernelSize,'conv10')
            out1_bn10 = batch_norm(out1_conv10,'bn10',bn_train=train_sign)
            out1_lrelu10 = lrelu(out1_bn10)
            
            result = conv2d(out1_lrelu10,1,kernelSize,'conv20')            
            
    return result + inputs
  
def pk_block(fk_1, p0, mask, lambd, mu):
    a_fk_1 = fft_tools.fft2d(fk_1) * mask
    fft = (tf.cast(lambd, tf.complex64) * (p0 - a_fk_1)) / tf.cast((1 + lambd + mu), tf.complex64)
#    fft = p0 - a_fk_1

    pk = fft_tools.ifft2d(fft)
    return tf.real(pk)

def fk_block(inputs, dropout_rate, train_sign=True):
    basicFilterNumScale = 4
    kernelSize = [3, 3]
    with tf.name_scope('fk_block'):
        with tf.variable_scope('fk_block'):
            
            out1_conv1 = conv2d(inputs,basicFilterNumScale,kernelSize,'conv1')
            out1_bn1 = batch_norm(out1_conv1,'bn1',bn_train=train_sign)
            out1_lrelu1 = lrelu(out1_bn1)
            
            out1_conv2 = conv2d(out1_lrelu1,basicFilterNumScale,kernelSize,'conv2')
            out1_bn2 = batch_norm(out1_conv2,'bn2',bn_train=train_sign)
            out1_lrelu2 = lrelu(out1_bn2)
            
            maxpool_1 = tf.nn.max_pool(out1_lrelu2,[1,2,2,1],[1,2,2,1],'VALID')
#            out1_conv3 = conv2d(maxpool_1,2*basicFilterNumScale,kernelSize,'conv3')
#            out1_bn3 = batch_norm(out1_conv3,'bn3',bn_train=train_sign)
#            out1_lrelu3 = lrelu(out1_bn3)
#            out1_conv4 = conv2d(out1_lrelu3,2*basicFilterNumScale,kernelSize,'conv4')
#            out1_bn4 = batch_norm(out1_conv4,'bn4',bn_train=train_sign)
#            out1_lrelu4 = lrelu(out1_bn4)
            
            out1_lrelu4 = Transformer.TransformerMoudle(maxpool_1, dropout_rate, train_sign, 'fk')
            
            upsamp1 = conv2d_transpose(out1_lrelu4,basicFilterNumScale,kernelSize,'up1')
            concat1 = tf.concat([out1_lrelu2,upsamp1],3)
            
            out1_conv5 = conv2d(concat1,basicFilterNumScale,kernelSize,'conv5')
            out1_bn5 = batch_norm(out1_conv5,'bn5',bn_train=train_sign)
            out1_lrelu5 = lrelu(out1_bn5)
            out1_conv6 = conv2d(out1_lrelu5,basicFilterNumScale,kernelSize,'conv6')
            out1_bn6 = batch_norm(out1_conv6,'bn6',bn_train=train_sign)
            out1_lrelu6 = lrelu(out1_bn6)
            
            result = conv2d(out1_lrelu6,inputs.shape[-1],kernelSize,'conv20')           
 
    return tf.nn.relu(result + inputs)
  
def iter_blcok(fk_1, p0, mask, dropout_rate=1., name='iter', train_sign=True, reuse=None):    
    with tf.name_scope(name):
        with tf.variable_scope(name, reuse=reuse):
            lambd = tf.get_variable(name='lambda', shape=[1], initializer=tf.constant_initializer(1))
            mu = tf.get_variable(name='mu', shape=[1], initializer=tf.constant_initializer(0))
            pk = pk_block(fk_1, p0, mask, lambd, mu)
            uk = uk_block(pk, train_sign=train_sign)
            fk = fk_block(fk_1 + ((1 + mu) / lambd) * uk, dropout_rate, train_sign)
#            fk = fk_block(fk_1 + uk, train_sign)

            return fk
    
    
def TRANSNet(f0, mask, p0, dropout_rate, train_sign=True, reuse=None):
    f1 = iter_blcok(f0, p0, mask, dropout_rate, 'iter_1', train_sign, reuse=reuse)
    f2 = iter_blcok(f1, p0, mask, dropout_rate, 'iter_2', train_sign, reuse=reuse)
    f3 = iter_blcok(f2, p0, mask, dropout_rate, 'iter_3', train_sign, reuse=reuse)        
    f4 = iter_blcok(f3, p0, mask, dropout_rate, 'iter_4', train_sign, reuse=reuse)        
    f5 = iter_blcok(f4, p0, mask, dropout_rate, 'iter_5', train_sign, reuse=reuse)        
    f6 = iter_blcok(f5, p0, mask, dropout_rate, 'iter_6', train_sign, reuse=reuse)        
    f7 = iter_blcok(f6, p0, mask, dropout_rate, 'iter_7', train_sign, reuse=reuse)        
    f8 = iter_blcok(f7, p0, mask, dropout_rate, 'iter_8', train_sign, reuse=reuse)
    f9 = iter_blcok(f8, p0, mask, dropout_rate, 'iter_9', train_sign, reuse=reuse)
    f10 = iter_blcok(f9, p0, mask, dropout_rate, 'iter_10', train_sign, reuse=reuse)

    return f10
    
