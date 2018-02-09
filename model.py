# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 22:45:24 2018

@author: Sandalfon
"""

import tensorflow as tf


class Model(object):

    @staticmethod
    def inference(x, drop_rate, filter1=48, ks1=5, ps1=2, std1=2, dl=7, dn=11):
        with tf.variable_scope('hidden1'):
            conv = tf.layers.conv2d(x, filter1=48, kernel_size=[ks1, ks1], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[ps1, ps1], strides=std1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden1 = dropout
        
        flat1 = 4 * 4 *filter1
        
        flatten = tf.reshape(hidden1, [-1, flat1])
        
        with tf.variable_scope('hidden2'):
            dense = tf.layers.dense(flatten, units=flat1, activation=tf.nn.relu)
            hidden9 = dense
            
        with tf.variable_scope('digit_length'):
            dense = tf.layers.dense(hidden2, units=dl)
            length = dense

        with tf.variable_scope('digit1'):
            dense = tf.layers.dense(hidden2, units=dn)
            digit1 = dense

        with tf.variable_scope('digit2'):
            dense = tf.layers.dense(hidden2, units=dn)
            digit2 = dense

        with tf.variable_scope('digit3'):
            dense = tf.layers.dense(hidden2, units=dn)
            digit3 = dense

        with tf.variable_scope('digit4'):
            dense = tf.layers.dense(hidden2, units=dn)
            digit4 = dense

        with tf.variable_scope('digit5'):
            dense = tf.layers.dense(hidden2, units=dn)
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
        