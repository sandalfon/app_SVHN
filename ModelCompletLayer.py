# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:28:02 2018

@author: Sandalfon
"""

import tensorflow as tf

"""
8 hiden layer 
1 Locally connected
2 dense connected
    1st hiden Maxout unit (3 units/filter), rest rectifier unites 
    units 48, 64, 128, 160 and 192 rest, 3072 for fully connected
    [fil_1, fil_2, fil_3,fil_4, fil_5, fil_fc ]
    Max pooling (2x2) and substractive norm (3x3)  mp, sn
    stride altern 2,1,2,... stri_odd, strid_even
    0 padding ("same")
    conv kernel = 5 (conv_kern)
    dropout all hidden
"""
class Model(object):

    @staticmethod
    def inference(x, model_options = {
            'conv_kern': 5, 
            'fil_1': 48, 
            'fil_2': 64, 
            'fil_3': 128, 
            'fil_4': 160, 
            'fil_5': 192, 
            'fil_fc': 3072,
            'fl_l': 7, 
            'fl_dig': 11,
            'stride_odd': 2, 
            'stride_even': 1,
            'mp': 2,
            'drop_rate': 0.2, 
            'mu': 0, 
            'sigma': 0.1
            }):
        conv_kern = model_options['conv_kern']
        fil_1 = model_options['fil_1']
        fil_2 = model_options['fil_2']
        fil_3 = model_options['fil_2']
        fil_4 = model_options['fil_4']
        fil_5 = model_options['fil_4']
        fil_fc = model_options['fil_fc']
        fl_l = model_options['fl_l']
        fl_dig = model_options['fl_dig']
        stride_odd = model_options['stride_odd']
        stride_even = model_options['stride_even']
        mp = model_options['mp']
        drop_rate = model_options['drop_rate']
        mu = model_options['mu']
        sigma = model_options['sigma']
        #Layer 1
        conv = tf.layers.conv2d(x, filters=fil_1, kernel_size=[conv_kern, conv_kern], padding='same')
        norm = tf.layers.batch_normalization(conv)
        activation = tf.nn.relu(norm)
        pool = tf.layers.max_pooling2d(activation, pool_size=[mp, mp], strides=stride_odd, padding='same')
        dropout = tf.layers.dropout(pool, rate=drop_rate)
        layer1 = dropout

        #Layer 2
        conv = tf.layers.conv2d(layer1, filters=fil_2, kernel_size=[conv_kern, conv_kern], padding='same')
        norm = tf.layers.batch_normalization(conv)
        activation = tf.nn.relu(norm)
        pool = tf.layers.max_pooling2d(activation, pool_size=[mp, mp], strides=stride_even, padding='same')
        dropout = tf.layers.dropout(pool, rate=drop_rate)
        layer2 = dropout
        
        #Layer 3
        conv = tf.layers.conv2d(layer2, filters=fil_3, kernel_size=[conv_kern, conv_kern], padding='same')
        norm = tf.layers.batch_normalization(conv)
        activation = tf.nn.relu(norm)
        pool = tf.layers.max_pooling2d(activation, pool_size=[mp, mp], strides=stride_odd, padding='same')
        dropout = tf.layers.dropout(pool, rate=drop_rate)
        layer3 = dropout
        
        #Layer 4
        conv = tf.layers.conv2d(layer3, filters=fil_4, kernel_size=[conv_kern, conv_kern], padding='same')
        norm = tf.layers.batch_normalization(conv)
        activation = tf.nn.relu(norm)
        pool = tf.layers.max_pooling2d(activation, pool_size=[mp, mp], strides=stride_even, padding='same')
        dropout = tf.layers.dropout(pool, rate=drop_rate)
        layer4 = dropout
        
        #Layer 5
        conv = tf.layers.conv2d(layer4, filters=fil_5, kernel_size=[conv_kern, conv_kern], padding='same')
        norm = tf.layers.batch_normalization(conv)
        activation = tf.nn.relu(norm)
        pool = tf.layers.max_pooling2d(activation, pool_size=[mp, mp], strides=stride_odd, padding='same')
        dropout = tf.layers.dropout(pool, rate=drop_rate)
        layer5 = dropout
        
        #Layer 6
        conv = tf.layers.conv2d(layer5, filters=fil_5, kernel_size=[conv_kern, conv_kern], padding='same')
        norm = tf.layers.batch_normalization(conv)
        activation = tf.nn.relu(norm)
        pool = tf.layers.max_pooling2d(activation, pool_size=[mp, mp], strides=stride_even, padding='same')
        dropout = tf.layers.dropout(pool, rate=drop_rate)
        layer6 = dropout
        
        #Layer 7
        conv = tf.layers.conv2d(layer6, filters=fil_5, kernel_size=[conv_kern, conv_kern], padding='same')
        norm = tf.layers.batch_normalization(conv)
        activation = tf.nn.relu(norm)
        pool = tf.layers.max_pooling2d(activation, pool_size=[mp, mp], strides=stride_odd, padding='same')
        dropout = tf.layers.dropout(pool, rate=drop_rate)
        layer7 = dropout
        
        #Layer 8
        conv = tf.layers.conv2d(layer7, filters=fil_5, kernel_size=[conv_kern, conv_kern], padding='same')
        norm = tf.layers.batch_normalization(conv)
        activation = tf.nn.relu(norm)
        pool = tf.layers.max_pooling2d(activation, pool_size=[mp, mp], strides=stride_even, padding='same')
        dropout = tf.layers.dropout(pool, rate=drop_rate)
        layer8 = dropout        
        
        #Layer 9
        #reshape 
        flatten = tf.reshape(layer8, [-1, 4 * 4 * fil_5])
        layer9 = tf.layers.dense(flatten, units = fil_fc, activation=tf.nn.relu)
        
        #Layer 10
        layer10 = tf.layers.dense(layer9, units = fil_fc, activation=tf.nn.relu)
        
        #layer out
        dense = tf.layers.dense(layer10, units=fl_l)
        length = dense

        dense = tf.layers.dense(layer10, units=fl_dig)
        digit1 = dense

        dense = tf.layers.dense(layer10, units=fl_dig)
        digit2 = dense

        dense = tf.layers.dense(layer10, units=fl_dig)
        digit3 = dense

        dense = tf.layers.dense(layer10, units=fl_dig)
        digit4 = dense

        dense = tf.layers.dense(layer10, units=fl_dig)
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