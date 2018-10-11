#coding: utf-8


"""
@author: Xilosopher
@contact: liyuhuangjojo@163.com
@Software: PyCharm
@file: sparse_placeholder.py
@time: 2018/7/7 22:21
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np


x = tf.sparse_placeholder(tf.float32)
y = tf.sparse_reduce_sum(x)

with tf.Session() as sess:
    indices = np.array([[3, 2, 0], [4, 5, 1]], dtype=np.int64)
    values = np.array([1.0, 2.0], dtype=np.float32)
    shape = np.array([7, 9, 2], dtype=np.int64)
    print(sess.run(y, feed_dict={x:tf.SparseTensorValue(indices, values, shape)}))
    print(sess.run(y, feed_dict={x:(indices, values, shape)}))
    sp = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    sp_value = sp.eval()
    print(sess.run(y, feed_dict={x: sp_value}))


