# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 12:10:21 2018

@author: Sandalfon
"""
import os
import numpy as np
from PIL import Image
class DataSet(object):
    """crop images and save them to numpy ndarray"""
    
    def __init__(self, digitStruct, folder):
        self.digitStruct = digitStruct
        self.folder = folder
        
    def setDataset(self):
        self.dataset = np.ndarray(shape=(len(self.digitStruct), 64, 64), dtype='float32')
        
        # 1 length + 5 labels, 0 represents none
        self.labels = np.ones(shape=(len(self.digitStruct), 6), dtype='int') * 10 
        
    def getDataset(self):
        self.setDataset()
        toRemove = []
        for i in range(len(self.digitStruct)):
            if (i % 500 == 0):
                 print('struct %d: %d' % (i, len(self.digitStruct)))
            img_file = os.path.join(self.folder,self.digitStruct[i]['filename'])
            img = Image.open(img_file)
            
            boxes = self.digitStruct[i]['boxes']
            if len(boxes)>5 :
                print(img_file,"size >5")
                toRemove.append(i)
            else:
                self.labels[i,0] = len(boxes)
                self.labels[i,1:len(boxes)+1] = [int(j['label']) for j in boxes]
            left = [j['left'] for j in boxes]
            top = [j['top'] for j in boxes]
            height = [j['height'] for j in boxes]
            width = [j['width'] for j in boxes]
            
            box = self.img_box(img, left, top, height, width)

            size = (64, 64)
            region = img.crop(box).resize(size)
            region = self.normalization(region)
            self.dataset[i,:,:] = region[:,:]
            
        print('dataset:',self.dataset.shape)
        print('labels:',self.labels.shape)
        return self.dataset, self.labels, toRemove
    
    def img_box(self, img, lefts, tops, heights, widths):
        img_ori_left = min(lefts)
        img_ori_top = min(tops)
        img_width = max(lefts) + max(widths) - img_ori_left
        img_height = max(tops) + max(heights) - img_ori_top
        
        img_left = img_ori_left * 0.95
        img_top = img_ori_top * 0.95
        #stay inside original image size
        img_right = min(img.size[0],(img_width + img_left)*1.05)
        img_bot = min(img.size[1],(img_height + img_top)*1.05)
        return (img_left, img_top, img_right,  img_bot)
        
    def normalization(self, img):
        img_norm = self.rgb2gray(img) # RGB to greyscale
        mean = np.mean(img_norm, dtype='float32')
        std = np.std(img_norm, dtype='float32', ddof=1)
        return (img_norm - mean) / std
    
    def rgb2gray(self, img):
        return np.dot(np.array(img, dtype='float32'), [0.299, 0.587, 0.114])