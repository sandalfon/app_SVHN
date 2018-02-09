# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 12:42:11 2018

@author: Sandalfon
"""

# These are all the modules we'll be using later.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

import os
import sys
import tarfile
import random

from PIL import Image
from IPython.display import display, Image
from six.moves import cPickle as pickle
from six.moves import range
from six.moves.urllib.request import urlretrieve
from collections import Counter
#%matplotlib inline

class Preprocess(object):
    
    def __init__(self):
        self.url = 'http://ufldl.stanford.edu/housenumbers/'
        self.last_percent_reported = None
        self.filename = 'train.tar.gz'
        self.expected_bytes = 404141560
        self.output_dir ='..\\data'
        
    def getCompletPath(self):
        return self.output_dir +'\\' + self.filename
    
    def download_progress_hook(self, count, blockSize, totalSize):
        """A hook to report the progress of a download. This is mostly intended for users with
        slow internet connections. Reports every 1% change in download progress.
        """
        #global last_percent_reported
        percent = int(count * blockSize * 100 / totalSize)
    
        if self.last_percent_reported != percent:
            if percent % 5 == 0:
                sys.stdout.write("%s%%" % percent)
                sys.stdout.flush()
            else:
                sys.stdout.write(".")
                sys.stdout.flush()
    
            self.last_percent_reported = percent
    
    def maybe_download(self, force=False):
        """Download a file if not present, and make sure it's the right size."""
        complet_path = None
        if force or not os.path.exists(self.getCompletPath()):
            print('Attempting to download:', self.filename) 
            complet_path, _ = urlretrieve(self.url + self.filename, self.getCompletPath(), reporthook=self.download_progress_hook)
            print('\nDownload Complete!')
        statinfo = os.stat(self.getCompletPath())
        if statinfo.st_size == self.expected_bytes:
            print('Found and verified', self.getCompletPath())
        else:
            raise Exception(
              'Failed to verify ' + self.filename + '. Can you get to it with a browser?')
        return complet_path
    
    def maybe_extract(self, force=False):
        root = os.path.splitext(os.path.splitext(self.getCompletPath())[0])[0]  # remove .tar.gz
        if os.path.isdir(root) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping extraction of %s.' % (root, self.getCompletPath()))
        else:
            print('Extracting data for %s. This may take a while. Please wait.' % root)
            tar = tarfile.open(self.getCompletPath())
            sys.stdout.flush()
            tar.extractall(self.output_dir)
            tar.close()
        if not os.path.exists(root+'/digitStruct.mat'):
            print("digitStruct.mat is missing")
        return root + '/digitStruct.mat'
    
    def save_pickles(self, pickle_file, train, test):
        try:
            f = open(pickle_file, 'wb')
            save = {
            'train_data': train,
            'test_data': test,
            }
            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise

    def load_pickles(self, pickle_file):
        train_data = {}
        test_data = {}
        with open(pickle_file, 'rb') as f:
            save = pickle.load(f)
            train_data = save['train_data']
            test_data = save['test_data']
            del save
            print('Training set', len(train_data))
            print('Test set', len(test_data))
        return train_data,test_data
#train_filename = maybe_download('train.tar.gz', 404141560 )
#test_filename = maybe_download('test.tar.gz', 276555967 )