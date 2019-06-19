#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 22:23:43 2019

@author: Deechean
"""

import pickle
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import time 
from PIL import Image

def load(file_name):
    with open(file_name, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
        return data  

class cifar10(object): 
    def __init__(self,path='cifar-10-batches-py/'):
        self.train_indexs = list()
        self.test_indexs = list()
        self.data_path = path
        self.train_images, self.train_labels = self._get_train()
        self.test_images, self.test_labels = self._get_test()
        self.label_dic = {0:'aircraft', 1:'car',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}

        
    def _get_train(self):
        train_labels = []        
        data1 = load(self.data_path+'data_batch_1')
        x1 = np.array(data1[b'data'])
        y1 = data1[b'labels']
        train_data = np.array(x1)
        train_labels = np.array(y1)
        
        data2 = load(self.data_path+'data_batch_2')
        x2 = np.array(data2[b'data'])
        y2 = data2[b'labels']
        train_data = np.append(train_data, x2)
        train_labels = np.append(train_labels, y2)

        data3 = load(self.data_path+'data_batch_3')
        x3 = np.array(data3[b'data'])
        y3 = np.array(data3[b'labels']).reshape(10000)
        train_data = np.append(train_data, x3)
        train_labels = np.append(train_labels, y3)

        data4 = load(self.data_path+'data_batch_4')
        x4 = np.array(data4[b'data'])
        y4 = np.array(data4[b'labels']).reshape(10000)
        train_data = np.append(train_data, x4)
        train_labels = np.append(train_labels, y4)
        
        data5 = load(self.data_path+'data_batch_5')
        x5 = np.array(data4[b'data'])
        y5 = np.array(data4[b'labels']).reshape(10000)
        train_data = np.append(train_data, x5)
        train_labels = np.append(train_labels, y5)
        
        train_data = train_data.reshape(-1, 3, 32, 32)
        train_labels.astype(np.int64)
        
        #for item in labels:
        #    train_labels.append(item)
        #print('image shape:',np.shape(train_data))
        #print('label shape:',np.shape(train_labels))  
        if len(train_data) != len(train_labels):
            assert('train images ' + str(len(train_data))+' doesnt equal to train labels' + str(len(train_labels)))
            
        print('train set length: '+str(len(train_data)))
        return train_data, train_labels
 
    def _get_test(self):
        test_labels = list()
        data1 = load(self.data_path+'test_batch')
        x = np.array(data1[b'data']).reshape(-1, 3, 32, 32)
        y = data1[b'labels']
        
        for item in y:
            test_labels.append(item)
        #print('test image shape:',np.shape(x))
        #print('test label shape:',np.shape(test_labels))        
        print('test set length: '+str(len(x)))
        return x, test_labels
    
    def _resize(self,image):
        resized_image = np.ndarray.reshape(image,(32,32,3))[2:30,2:30,0:3] 
        #print(resized_image.shape)
        return resized_image
    
    def random_flipper(self,image):
        if random.random() < 0.5:
            swap_time = int(len(image[0])/2)
            for i in range(swap_time):
                image[0][[i,len(image[0])-i-1],:] = image[0][[len(image[0])-i-1,i],:]
                image[1][[i,len(image[1])-i-1],:] = image[1][[len(image[1])-i-1,i],:]
                image[2][[i,len(image[2])-i-1],:] = image[2][[len(image[2])-i-1,i],:]
        return image
        
    def image_distort(self,image):
        
        return image
    
    def random_bright(self, image, delta=32):
        if random.random() < 0.5:
            delta_r = int(random.uniform(-delta, delta))
            delta_g = int(random.uniform(-delta, delta))
            delta_b = int(random.uniform(-delta, delta))

            R = image[0] + delta_r
            G = image[1] + delta_g
            B = image[2] + delta_b
            
            image = np.asarray([R,G,B])
            #print(2)
            #print(np.shape(image))
            image = image.clip(min=0, max=255)
        return image
   
    def get_train_batch(self,batch_size=128, augument = True, new_size=(32,32), resize = False):
        batch_image = list()
        batch_label = list()
        data_index = list()
        i = 0
        while i < batch_size:
            index = random.randint(0, len(self.train_labels)-1)
            if not index in self.train_indexs:
                i += 1
                d = self.train_images[index]
                if augument:
                    d = self.random_bright(self.random_flipper(d))
                if resize:
                    d = self.resize_image(d,new_size)
                batch_image.append(d)
                batch_label.append(self.train_labels[index])
                self.train_indexs.append(index)
                data_index.append(index)
                if len(self.train_indexs) >=  len(self.train_images):
                    self.train_indexs.clear()
        return np.array(batch_image).transpose(0,3,2,1).reshape(-1,new_size[0],new_size[1],3), batch_label, data_index
        
    def get_test_batch(self,batch_size=10000, new_size=(32,32), resize = False):
        batch_image = list()
        batch_label = list()
        data_index = list()
        i = 0
        while i < batch_size:
            index = random.randint(0, len(self.test_labels)-1)
            if not index in self.test_indexs:
                i += 1
                d = self.test_images[index]
                if resize:
                    d = self.resize_image(d,new_size)
                batch_image.append(d) 
                batch_label.append(self.test_labels[index])
                self.test_indexs.append(index)
                data_index.append(index)
                if len(self.test_indexs) >=  len(self.test_images):
                    self.test_indexs.clear()
        return  np.array(batch_image).transpose(0,3,2,1).reshape(-1,new_size[0],new_size[1],3), batch_label,data_index
    
    def resize_image(self, image, new_size):
        img = Image.fromarray(image.astype('uint8').transpose(2,1,0))       
        return np.asarray(img.resize(new_size,resample=Image.BILINEAR)).transpose(1,2,0)          
        