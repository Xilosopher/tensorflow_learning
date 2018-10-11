# encoding: utf-8


"""
@author: Xilosopher
@contact: liyuhuangjojo@163.com
@software: PyCharm
@file: tensor.py
@time: 2018/7/2 16:33
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf


a = tf.constant([1, 1])
b = tf.constant([2, 2])
c = tf.add(a, b)

with tf.Session() as sess:
    print("a[0] = %s, a[1] = %s" % (a[0].eval(), a[1].eval()))
    print("c.name = %s" % c.name)
    print("c.value = %s" % c.eval())
    print("c.shape = %s" % c.shape)
    print("a.consumers = %s" % a.consumers())
    print("b.consumers = %s" % b.consumers())
    print("[c.op]: \n%s" % c.op)


