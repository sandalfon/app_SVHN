# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:28:22 2018

@author: Sandalfon
"""
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def disp_sample_dataset(dataset, label):
    items = random.sample(range(dataset.shape[0]), 8)
    for i, item in enumerate(items):
        plt.subplot(2, 4, i+1)
        plt.axis('off')
        plt.title(str(label[item][0]) + str(label[item][1:5]))
        plt.imshow(dataset[item,:,:])                   

def accuracy_single(predictions, labels):
    """calculate character-level accuracy"""
    a = np.argmax(predictions, 2).T == labels[:,1:6]
    length = labels[:,0]
    summ = 0.0
    for i in range(len(length)):
        summ += np.sum(a[i,:length[i]])
    return(100 * summ / np.sum(length))
    
def accuracy_multi(predictions, labels):
    """calculate sequence-level accuracy"""
    count = predictions.shape[1]
    return 100.0 * (count - np.sum([1 for i in np.argmax(predictions, 2).T == labels[:,1:6] if False in i])) / count


def build_batch(data_image, data_label, batch_size, shuffled):
    length = np.count_nonzero(np.less(data_label,9),axis = 1)
    length = tf.cast(length, tf.int32)
    digits = data_label
    num_examples = data_label.shape[0]
    min_queue_examples = int(0.4 * num_examples)
    if shuffled:
        image_batch, length_batch, digits_batch = tf.train.shuffle_batch([data_image, length, digits],
         batch_size=batch_size,
         num_threads=2,
         capacity=min_queue_examples + 3 * batch_size,
         min_after_dequeue=min_queue_examples)
    else:
        image_batch, length_batch, digits_batch = tf.train.batch([data_image, length, digits],
         batch_size=batch_size,
         num_threads=2,
         capacity=min_queue_examples + 3 * batch_size)
    return image_batch, length_batch, digits_batch