#coding: utf-8


"""
@author: Moro_JoJo
@contact: liyuhuangjojo@163.com
@file: save_pic.py
@time: 2018/5/13 18:03
"""


from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("------------ Train Data -------------")
print(mnist.train.images.shape)
print(mnist.train.labels.shape)

print("------------ Validation Data -------------")
print(mnist.validation.images.shape)
print(mnist.validation.labels.shape)

print("------------ Test Data -------------")
print(mnist.test.images.shape)
print(mnist.test.labels.shape)

print("------------ Show Some Data -------------")
print(mnist.train.images[0, :])


