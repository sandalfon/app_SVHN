# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 23:16:51 2018

@author: Sandalfon
"""
from preprocess_data import Preprocess
from digitStructFile import DigitStructFile
from dataSet import DataSet
from dataRefine import DataRefine
import matplotlib.pyplot as plt
import utils

preprocess = Preprocess()
preprocess.url = 'http://ufldl.stanford.edu/housenumbers/'
preprocess.last_percent_reported = None
preprocess.filename = 'train.tar.gz'
preprocess.expected_bytes = 404141560
preprocess.output_dir ='..\\data'
preprocess.maybe_download()
preprocess.maybe_extract()

preprocess.filename = 'test.tar.gz'
preprocess.expected_bytes = 276555967
preprocess.maybe_download()
preprocess.maybe_extract()


train = DigitStructFile('..\\data\\train\\digitStruct.mat')
train_data = train.getAllDigitStructure_ByDigit()
train_data[:10]

test = DigitStructFile('..\\data\\test\\digitStruct.mat')
test_data = test.getAllDigitStructure_ByDigit()
test_data[0]


pickle_file = '..\\data\\multi_bbox_info.pickle'
preprocess.save_pickles(pickle_file, train_data, test_data)
train_data, test_data = preprocess.load_pickles(pickle_file)




print('Training set', len(train_data))
print('Test set', len(test_data))
dataR = DataRefine()
dataR.make_from_dataset(train_data, test_data, "..\\data")

dataR.save_pickles("..\\data\\multi_crop.pickle")



dataR2=DataRefine()
dataR2.load_pickles("..\\data\\multi_crop.pickle")
print(dataR2.test_labels.shape)
print('Training set', dataR2.train_dataset.shape, dataR2.train_labels.shape)
print('Validation set', dataR2.valid_dataset.shape, dataR2.valid_labels.shape)
print('Test set', dataR2.test_dataset.shape, dataR2.test_labels.shape)





import utils
image_batch, length_batch, digits_batch = utils.build_batch(
        dataR2.train_dataset, dataR2.train_labels, 32, True)
print(length_batch.shape)
