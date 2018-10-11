#coding: utf-8


"""
@author: Xilosopher
@contact: liyuhuangjojo@163.com
@Software: PyCharm
@file: placeholder.py
@time: 2018/7/7 21:46
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np


x = tf.placeholder(dtype=tf.float32, shape=(2, 2), name="x")
y = tf.matmul(x, x, name="matmul")

with tf.Session() as sess:
    rand_array = np.random.rand(2, 2)
    print(sess.run(y, feed_dict={x:rand_array}))

