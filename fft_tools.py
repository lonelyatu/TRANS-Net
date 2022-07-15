# -*- coding: utf-8 -*-


import tensorflow as tf

def fft2d(inputs):
    inputs_perm = tf.transpose(inputs, [0, 3, 1, 2])
    inputs_complex = tf.cast(inputs_perm, tf.complex64)
    inputs_fft = tf.signal.fft2d(inputs_complex)
    inputs_fftshift = tf.signal.fftshift(inputs_fft, (2,3))
    inputs_fftshift_perm = tf.transpose(inputs_fftshift, [0, 2, 3, 1])
    return inputs_fftshift_perm

def ifft2d(inputs):
    inputs_perm = tf.transpose(inputs, [0, 3, 1, 2])
    inputs_ifftshift = tf.signal.ifftshift(inputs_perm, (2,3))
    inputs_ifft = tf.signal.ifft2d(inputs_ifftshift)
    inputs_ifft_perm = tf.transpose(inputs_ifft, [0, 2, 3, 1])
    return inputs_ifft_perm

