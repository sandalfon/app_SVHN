# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:27:31 2018

@author: Sandalfon
"""
import tensorflow as tf
from tensorflow.contrib.layers import flatten



class Model(object):

    @staticmethod
    def inference(x, num_channels = 1, depth_c1 = 6, depth_c3 = 16, 
                 depth_c5 = 120, depth_f6 = 84, depth_fout = 10, patch_c1 = 28, 
                 patch_c3 = 10, patch_c5 = 5, dR = 0.8, mu = 0, sigma = 0.1 ):
    
    
        #####C1 
        # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
        weight_c1  = tf.Variable(tf.truncated_normal([patch_c1, patch_c1, 
                                num_channels, depth_c1]),std_dev = sigma)
        biais_c1 = tf.Variable(tf.zeros(depth_c1))
        c1 = tf.nn.conv2d(x, weight_c1, [1, 1, 1, 1], padding = "SAME")
        
        #####S2
        # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
        s2 = tf.nn.dropout(tf.nn.max_pool(tf.nn.relu(c1 + biais_c1), 
                                ksize = [1, 2, 2, 1], 
                                strides = [1, 2, 2, 1],padding = "SAME"), dR)
        
        #####C3
        # TODO: Layer 2: Convolutional. Output = 10x10x16.
        weight_c3  = tf.Variable(tf.truncated_normal([patch_c3, patch_c3, 
                                depth_c1, depth_c3]),std_dev = sigma)
        biais_c3 = tf.Variable(tf.zeros(depth_c3))
        c3 = tf.nn.conv2d(s2, weight_c3, [1, 1, 1, 1], padding = "SAME")
        
        #####S4
        # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
        s4 = tf.nn.dropout(tf.nn.max_pool(tf.nn.relu(c3 + biais_c3), 
                                ksize = [1, 2, 2, 1], 
                                strides = [1, 2, 2, 1],padding = "SAME"), dR)
        
        ######C5
        # TODO: Flatten. Input = 5x5x16. Output = 400.
        c5 = flatten(s4)
        # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
        weight_c5 = tf.Variable(tf.truncated_normal(
                shape = (depth_c3*patch_c5*patch_c5, depth_c5), 
                mean = mu, stddev = sigma))
        biais_c5 = tf.Variable(tf.zeros(depth_c5))
        c5 = tf.nn.relu(tf.matmul(c5,weight_c5) + biais_c5)
        
        ######F6
        weight_f6 = tf.Variable(
                tf.truncated_normal(shape = (depth_c5,
                                             depth_f6), mean = mu, stddev = sigma))
        biais_f6 = tf.Variable(tf.zeros(depth_f6))
        f6 = tf.matmul(c5,weight_f6) + biais_f6
        # TODO: Activation.
        f6 = tf.nn.relu(f6)
        
        ######Foutput
        weight_fout = tf.Variable(
                tf.truncated_normal(shape = (depth_f6,
                                             depth_fout),
                                    mean = mu, stddev = sigma))
        biais_fout = tf.Variable(tf.zeros(depth_fout))
        f6 =  tf.matmul(f6, weight_fout) + biais_fout
 
        dense = tf.layers.dense(f6, units=7)
        length = dense

        dense = tf.layers.dense(f6, units=11)
        digit1 = dense

        dense = tf.layers.dense(f6, units=11)
        digit2 = dense

        dense = tf.layers.dense(f6, units=11)
        digit3 = dense

        dense = tf.layers.dense(f6, units=11)
        digit4 = dense

        dense = tf.layers.dense(f6, units=11)
        digit5 = dense

        length_logits, digits_logits = length, tf.stack([digit1, digit2, digit3, digit4, digit5], axis=1)
        return length_logits, digits_logits
    
    @staticmethod
    def loss(length_logits, digits_logits, length_labels, digits_labels):
        length_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=length_labels, logits=length_logits))
        digit1_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 0], logits=digits_logits[:, 0, :]))
        digit2_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 1], logits=digits_logits[:, 1, :]))
        digit3_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 2], logits=digits_logits[:, 2, :]))
        digit4_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 3], logits=digits_logits[:, 3, :]))
        digit5_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 4], logits=digits_logits[:, 4, :]))
        loss = length_cross_entropy + digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy + digit4_cross_entropy + digit5_cross_entropy
        return loss