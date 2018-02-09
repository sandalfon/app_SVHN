# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 14:31:29 2018

@author: Sandalfon
"""
from dataSet import DataSet
from six.moves import cPickle as pickle
import numpy as np

class DataRefine(object):
    
    def __init__(self):
        self.train_dataset = None
        self.train_labels = None
        self.valid_dataset = None
        self.valid_labels = None
        self.test_dataset = None
        self.test_labels = None
        
    def load_pickles(self, filename):
        with open(filename, 'rb') as f:
            save = pickle.load(f)
            self.train_dataset = save['train_dataset']
            self.train_labels = save['train_labels']
            self.valid_dataset = save['valid_dataset']
            self.valid_labels = save['valid_labels']
            self.test_dataset = save['test_dataset']
            self.test_labels = save['test_labels']
            del save  # hint to help gc free up memory
            print('Training set', self.train_dataset.shape, self.train_labels.shape)
            print('Validation set', self.valid_dataset.shape, self.valid_labels.shape)
            print('Test set',self. test_dataset.shape, self.test_labels.shape)
        
    def save_pickles(self, filename):
        try:
            f = open(filename, 'wb')
            save = {
            'train_dataset': self.train_dataset,
            'train_labels': self.train_labels,
            'valid_dataset': self.valid_dataset,
            'valid_labels': self.valid_labels,
            'test_dataset': self.test_dataset,
            'test_labels': self.test_labels,
            }
            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
            f.close()
            print("Done")
        except Exception as e:
            print('Unable to save data to', filename, ':', e)
            raise
        
    def make_from_dataset(self, train_data, test_data, img_path, train_portion=6000):
        train_dataset = DataSet(train_data, img_path + '\\train')
        train_data, train_labels, train_toRemove = train_dataset.getDataset()
        train_data = np.delete(train_data, train_toRemove, axis=0)
        train_labels = np.delete(train_labels, train_toRemove, axis=0)
        print(train_data.shape)
        print(train_labels.shape)
        
        
        test_dataset = DataSet(test_data, img_path + '\\test')
        test_data, test_labels, test_toRemove = test_dataset.getDataset()
        test_data = np.delete(test_data, test_toRemove, axis=0)
        test_labels = np.delete(test_labels, test_toRemove, axis=0)
        print(test_data.shape)
        print(test_labels.shape)
        
        train_data, train_labels = self.randomize(train_data, train_labels)
        test_data, test_labels = self.randomize(test_data, test_labels)
        
    
        self.valid_dataset = train_data[:train_portion,:,:]
        self.valid_labels = train_labels[:train_portion]
        self.train_dataset = train_data[train_portion:,:,:]
        self.train_labels = train_labels[train_portion:]
        self.test_dataset = test_data
        self.test_labels = test_labels  
                  
        print(self.train_dataset.shape, self.train_labels.shape)
        print(self.test_dataset.shape, self.test_labels.shape)
        print(self.valid_dataset.shape, self.valid_labels.shape)
        
        
    def randomize(self,dataset, labels):
        permutation = np.random.permutation(labels.shape[0])
        shuffled_dataset = dataset[permutation,:,:]
        shuffled_labels = labels[permutation]
        return shuffled_dataset, shuffled_labels
        
