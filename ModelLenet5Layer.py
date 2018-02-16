# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:27:31 2018

@author: Sandalfon
"""
import tensorflow as tf
from tensorflow.contrib.layers import flatten



class Model(object):

    @staticmethod
    def inference(x, ks_a1 = 5, ks_a3 = 5, ks_c5 = 5, 
                  fl_a1 = 48, fl_a3 = 64, fl_c1 = 128, fl_fc2 = 192, fl_l = 7,
                  fl_dig = 11, fl_out = 3072,
                  ps_a2 = 2, ps_a4 = 2,
                  drop_rate = 0.8, mu = 0, sigma = 0.1 ):
    
        
        ##### Layer 1 : CNN1 
        a1 = tf.layers.conv2d(x, filters=fl_a1, kernel_size=[ks_a1, ks_a1], 
                                padding='same')
        norm_c1 = tf.layers.batch_normalization(a1)
        cnn1 = tf.nn.relu(norm_c1)
        
        #####Layer 2 : pool1
        a2 = tf.layers.max_pooling2d(cnn1, pool_size=[ps_a2, ps_a2], 
                                       strides=2, padding='same')
        pool1 = tf.layers.dropout(a2, rate=drop_rate)
        

        #####Layer 3 : CNN2 
        a3 = tf.layers.conv2d(pool1, filters=fl_a3, kernel_size=[ks_a3, ks_a3], 
                                padding='same')
        norm_c3 = tf.layers.batch_normalization(a3)
        cnn2 = tf.nn.relu(norm_c3)
        
        #####Layer 4 : pool2
        a4 = tf.layers.max_pooling2d(cnn2, pool_size=[ps_a4, ps_a4], 
                                       strides=2, padding='same')
        pool2 = tf.layers.dropout(a4, rate=drop_rate)
        
        ######Layer 5 : fully connected 1 fc1
        fc1 = flatten(pool2, [-1, 4 * 4 * fl_c1])

        ############Layer 6 : fully connected 2 fc2
        fc2 = tf.layers.dense(fc1, units = fl_fc2, activation=tf.nn.relu)
        
        ######Layer output Foutput
        fout = tf.layers.dense(fc2, units = fl_out, activation=tf.nn.relu)
 
        dense = tf.layers.dense(fout, units=fl_l)
        length = dense

        dense = tf.layers.dense(fout, units=fl_dig)
        digit1 = dense

        dense = tf.layers.dense(fout, units=fl_dig)
        digit2 = dense

        dense = tf.layers.dense(fout, units=fl_dig)
        digit3 = dense

        dense = tf.layers.dense(fout, units=fl_dig)
        digit4 = dense

        dense = tf.layers.dense(fout, units=fl_dig)
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