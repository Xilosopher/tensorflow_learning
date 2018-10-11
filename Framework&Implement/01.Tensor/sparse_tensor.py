# encoding: utf-8


"""
@author: Xilosopher
@contact: liyuhuangjojo@163.com
@software: PyCharm
@file: sparse_tensor.py
@time: 2018/7/3 9:59
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

sp = tf.SparseTensor(indices=[[0, 2], [1, 3]], values=[1, 2], dense_shape=[3,4])
with tf.Session() as sess:
    print(sp.eval())

x = tf.SparseTensor(indices=[[0, 0], [0, 2], [1, 1]], values=[1, 1, 1], dense_shape=[2, 3])
reduce_x = [
    tf.sparse_reduce_sum(x),                            # => 3
    tf.sparse_reduce_sum(x, axis=1),                    # => [2, 1]
    tf.sparse_reduce_sum(x, axis=1, keep_dims=True),   # => [[2], [1]]
    tf.sparse_reduce_sum(x, axis=[0, 1])               # => 3
]
with tf.Session() as sess:
    print(sess.run(reduce_x))

