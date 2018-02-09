# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:28:22 2018

@author: Sandalfon
"""
import random
import matplotlib.pyplot as plt
import numpy as np

def disp_sample_dataset(dataset, label):
    items = random.sample(range(dataset.shape[0]), 8)
    for i, item in enumerate(items):
        plt.subplot(2, 4, i+1)
        plt.axis('off')
<<<<<<< HEAD
        plt.title(str(label[item][0]) + str(label[item][1:5]))
        plt.imshow(dataset[item,:,:])                   
=======
        plt.title(label[i][1:5])
        plt.imshow(dataset[i,:,:])                   
>>>>>>> 7cd155506495b6937f2edceae4db7865c4bcdb2b
        
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