#coding: utf-8


"""
@author: Moro_JoJo
@contact: liyuhuangjojo@163.com
@file: save_pic.py
@time: 2018/5/13 18:03
"""


from tensorflow.examples.tutorials.mnist import  input_data
import scipy.misc
import numpy as np
import os


mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
save_dir = 'MNIST_data/raw/'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)

for i in range(20):
    # mnist.images[i, :]表示第i张图片
    image_array = mnist.train.images[i, :]
    # MNIST图片是一个784维向量，需要还原为28x28的图像
    image_array = image_array.reshape(28, 28)
    # 保存文件
    file_name = save_dir + 'mnist_train_%d.jpg' % i
    scipy.misc.toimage(arr=image_array, cmin=0.0, cmax=1.0).save(file_name)

    # 标签
    one_hot_label = mnist.train.labels[i, :]
    # np.argmax可以直接获取原始label
    label = np.argmax(one_hot_label)
    print('mnist_train_%d.jpg label: %d' % (i, label))





