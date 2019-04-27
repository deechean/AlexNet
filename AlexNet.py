#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:28:29 2019

@author: Deechean
"""
import tensorflow as tf
import tf_general as tfg


class alexnet(object):

    def __init__(self, x, n_class=10, keep_prob=1.0):
        self.input = x
        self.n_class = n_class
        self.keep_prob = keep_prob
        self._build_net()

    def _build_net(self):
        with tf.name_scope('norm'):    
            self.x_norm = tf.nn.l2_normalize(tf.cast(self.input, tf.float32),dim=0)
            print('norm: ', self.x_norm.get_shape())
        
        with tf.name_scope('conv_1'):
            self.conv1 = tfg.conv2d(self.x_norm, 11, 4, 96, 'conv1', 'VALID')
            print('conv_1: ', self.conv1.get_shape())
            
        with tf.name_scope('pool_1'):
            self.pool1 = tfg.max_pool(self.conv1, 3, 2, 'pool1', 'VALID')
            print('pool_1: ', self.pool1.get_shape())
            
        with tf.name_scope('conv_2'):
            self.conv2 = tfg.conv2d(self.pool1, 5, 1, 256, 'conv2', 'SAME')
            print('conv_2: ', self.conv2.get_shape())
            
        with tf.name_scope('pool_2'):
            self.pool2 = tfg.avg_pool(self.conv2, 3, 2, 'pool2', 'VALID')
            print('pool_2: ', self.pool2.get_shape())
            
        with tf.name_scope('conv_3'):
            self.conv3 = tfg.conv2d(self.pool2, 3, 1, 384, 'conv3', 'SAME')
            print('conv_3: ', self.conv3.get_shape())

        with tf.name_scope('conv_4'):
            self.conv4 = tfg.conv2d(self.conv3, 3, 1, 384, 'conv4', 'SAME')
            print('conv_4: ', self.conv4.get_shape())
        
        with tf.name_scope('conv_5'):
            self.conv5 = tfg.conv2d(self.conv4, 3, 1, 256, 'conv5', 'SAME')
            print('conv_5: ', self.conv5.get_shape())
            
        with tf.name_scope('pool_3'):
            self.pool3 = tfg.max_pool(self.conv5, 3, 2, 'pool3', 'VALID')
            print('pool_3: ', self.pool3.get_shape())
            
        with tf.name_scope('flat_1'):
            self.flat1, self.flat_dim = tfg.flatten(self.pool3)
            print('flat_1: ', self.flat1.get_shape())         

        with tf.name_scope('fc_1'):
            self.fc1 = tfg.fc_layer(self.flat1, self.flat_dim, 4096, 'fc1')
            print('fc_1: ', self.fc1.get_shape())

        with tf.name_scope('fc_2'):
            self.fc2 = tfg.fc_layer(self.fc1, 4096, 4096, 'fc2')
            print('fc_2: ', self.fc2.get_shape())
        
        with tf.name_scope('fc_3'):
            self.fc3 = tfg.fc_layer(self.fc2, 4096, 10, 'fc3')
            print('fc_3: ', self.fc3.get_shape())

        with tf.name_scope('drop_out_1'):
            self.drop1 = tfg.drop_out(self.fc3, self.keep_prob, 'drop_out1')
            self.logits = self.drop1
            print('drop_out_1: ', self.drop1.get_shape())

        with tf.name_scope('prediction'):
            self.prediction = tf.nn.softmax(self.drop1)
            print('prediction: ', self.prediction.get_shape())
            