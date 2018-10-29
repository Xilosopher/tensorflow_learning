# encoding: utf-8


"""
@author: Xilosopher
@contact: liyuhuangjojo@163.com
@software: PyCharm
@file: experiment
@time: 2018/10/27 9:20
"""


import os
import tensorflow as tf
from tensorflow.python.client import device_lib
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "99"


def get_local_device_info():
    print(device_lib.list_local_devices())


def calculate_with_cpu():
    # 使用CPU进行计算
    with tf.device("/cpu:0"):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
        c = tf.matmul(a,b)
        # 查看计算时硬件的使用情况
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        print(sess.run(c))


def calculate_with_gpu():
    # 使用GPU进行计算
    with tf.device("/gpu:0"):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
        c = tf.matmul(a,b)
        # 查看计算时硬件的使用情况
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        print(sess.run(c))


if __name__ == '__main__':
    get_local_device_info()
    calculate_with_cpu()
    calculate_with_gpu()