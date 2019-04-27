#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 22:08:46 2019

@author: Deechean
"""
import cifar10
import os
import tensorflow as tf
from AlexNet import alexnet
import tf_general as tfg
import numpy as np


FLAGS = tf.flags.FLAGS
try:
    tf.flags.DEFINE_integer('epoch', 500, 'epoch')
    tf.flags.DEFINE_integer('batch_size', 128, 'batch size')
    tf.flags.DEFINE_integer('test_size', 5000, 'test size')
    tf.flags.DEFINE_float('lr', 0.01, 'learning rate')
    tf.flags.DEFINE_boolean('restore', False, 'restore from checkpoint and run test')
except:
    print('parameters have been defined.')

cifar10_dir='cifar-10-batches' 
#获取数据增强后的训练集数据
train_image, train_label = cifar10.distorted_inputs(cifar10_dir,FLAGS.batch_size)
#获取数据裁剪后的测试数据
test_image, test_label = cifar10.inputs(eval_data=True,data_dir=cifar10_dir
                                        ,batch_size=FLAGS.test_size)    
   
ckpt_dir = 'ckpt/'

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

with tf.name_scope('input'):
    x_image = train_image
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    y_ = train_label

with tf.name_scope('prediction'):
    alex_net = alexnet(x_image, keep_prob)
    y = alex_net.prediction

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,
                           labels=y_, name="cross_entropy_per_example")
    cross_entropy = tf.reduce_mean(cross_entropy)
    
    
with tf.name_scope('train_step'):
    train_step = tf.train.AdagradOptimizer(FLAGS.lr).minimize(cross_entropy)
    #train_step= tf.train.GradientDescentOptimizer(FLAGS.lr).minimize(cross_entropy)

saver=tf.train.Saver(max_to_keep = 5)

correct_prediction = tf.nn.in_top_k(y, y_, 1)
accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))/128

if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        for i in range(FLAGS.epoch):
            train_image_batch, train_label_batch = sess.run([train_image, train_label])
            loss, _ , output,accuracy_rate = sess.run([cross_entropy,train_step,y, accuracy], 
                                                   feed_dict={keep_prob: 0.5}) 
            print('.', end='')
            """
            log = []
            for j in range(FLAGS.batch_size):
                str_pre += str(fc_2[j])+'\n'
                str_pre += str(fcnorm_[j])+'\n'
                str_pre +=str(logits_[j])+'\n'
                str_pre +=str(np.round(prediction[j],decimals=2)) + '\n'
                str_pre +=str(labels[j]) + '\n'
                str_pre += str(logy_[j]) + '\n'
                str_pre += str(loss)+ '\n'                
            log.appendstr_pre
            tfg.saveEvalData('./cifar_log/iter'+str(i)+'_cifar.txt',log)
            """           
            if i % 10 == 0:  #保存预测模型
                print('')
                saver.save(sess,'ckpt/cifar10_'+str(i)+'.ckpt',global_step=i)  
                print('iter:' + str(i), str(round(accuracy_rate*100,2))+'%,  loss=' + str(loss))
        
        #le_net5.input = tf.reshape(test_image, [-1, 32, 32,3])
        accuracy_rate = sess.run(accuracy)
        print('Test accuracy:', str(round(accuracy_rate*100,2))+'%')
           
        tf.reset_default_graph()