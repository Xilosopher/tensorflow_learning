#coding: utf-8


"""
@author: Xilosopher
@contact: liyuhuangjojo@163.com
@Software: PyCharm
@file: session.py
@time: 2018/7/8 23:21
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
with tf.Session() as sess:
    print("Simple Calculation: ", sess.run(fetches=c))


x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x * y
with tf.Session() as sess:
    print("Dependent Calculation: ", sess.run(fetches=z, feed_dict={x: 3.0, y: 2.0}))


x = tf.placeholder(tf.float32)
W = tf.Variable(1.0)
b = tf.Variable(1.0)
y = W * x + b
with tf.Session() as sess:
    tf.global_variables_initializer().run()  # Operation.run
    fetch = y.eval(feed_dict={x: 3})         # Tensor.run
    print("With Session Calculation: ", fetch)


sess2 = tf.InteractiveSession()
print("Interactive Session: ", c.eval())
sess2.close()



