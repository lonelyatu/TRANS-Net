# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from scipy import io
import tensorflow as tf

from scipy import fftpack

def to_bad_image(x, mask):
    fft = fftpack.fft2(x)
    fftshift = fftpack.fftshift(fft) * mask
    ifftshift = fftpack.ifftshift(fftshift)
    img = fftpack.ifft2(ifftshift)
    return np.abs(img)   

def scale_to_01(x):
    minV = x.min()
    maxV = x.max()
    return (x - minV) / (maxV - minV)    

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def TFRecordFileGen(mask):
    
    lst = glob.glob('./TrainFile/*.npy')
    if not os.path.exists('./TFRecord'):
        os.mkdir('./TFRecord')
    for j in range(1000):
        name = './TFRecord/train_%d.tfrecord' % j
        writer = tf.python_io.TFRecordWriter(name)
        label = scale_to_01(np.float32(np.load(lst[j])))

        img = to_bad_image(label, mask)
        print(j)
        img_raw = img.tostring()
        lab_raw = label.tostring()
#                print(np.square(img-label).mean())
        example = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(img_raw),        
        'res': _bytes_feature(lab_raw),
        }))
        writer.write(example.SerializeToString())
				
        writer.close()

def read_and_decode(filename_queue, shape_input,shape_label):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'res': tf.FixedLenFeature([], tf.string),
                })

    image = tf.decode_raw(features['image'], tf.float32, little_endian=True)
    image = tf.reshape(image,[shape_input[0],shape_input[1],shape_input[2]])
    
    res = tf.decode_raw(features['res'], tf.float32, little_endian=True)
    res = tf.reshape(res,[shape_label[0],shape_label[1],shape_label[2]])
    return image, res


def input_pipeline(filenames, batch_size, num_epochs=None, 
                   num_features_input=None,num_features_label=None):
    '''num_features := width * height for 2D image'''
    filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs, shuffle=True)
    example, label = read_and_decode(filename_queue, shape_input=num_features_input,
                                     shape_label=num_features_label)
#    label = read_and_decode(filename_queue, num_features)
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 40000 // 10
    capacity = min_after_dequeue + 10 * batch_size 
    img_batch, res_batch = tf.train.shuffle_batch(
            [example, label], batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue,num_threads=64,allow_smaller_final_batch=True)
    return img_batch, res_batch

def parse_function(example_proto):
    features = tf.parse_single_example(
            example_proto,
            features={
                'res': tf.FixedLenFeature([], tf.int64),
                'image': tf.FixedLenFeature([], tf.string),
                })

    image = tf.decode_raw(features['image'], tf.float32, little_endian=True)
    image = tf.reshape(image,[64,64])
    label = features['res']
    return image, label

if __name__ == '__main__':
    mask = io.loadmat('./GaussianDistribution2DMask_10.mat')['maskRS2']
    TFRecordFileGen(mask)
    print("convert finishes")
    #read_test()







