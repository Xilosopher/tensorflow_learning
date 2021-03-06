#coding: utf-8


"""
@author: Xilosopher
@contact: liyuhuangjojo@163.com
@Software: PyCharm
@file: normal_way.py
@time: 2018/7/29 13:36
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf


#####################################  Path configuration  ###################################################
root_path = './data/'
if not os.path.exists(root_path):
    os.makedirs(root_path)
data_path = root_path + 'stat.tfrecord'


###############################  Write Data To TFRecords File ################################################
# Create writer
writer = tf.python_io.TFRecordWriter(data_path)
# Make example data in loop
for i in range(1, 3):
    # Create examples defined in example.proto
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
                'age': tf.train.Feature(int64_list=tf.train.Int64List(value=[i * 24])),
                'income': tf.train.Feature(float_list=tf.train.FloatList(value=[i * 2048.0])),
                'outgo': tf.train.Feature(float_list=tf.train.FloatList(value=[i * 1024.0]))
            }
        )
    )
    # Serialize example data to strings, and write strings into stat.tfrecord
    writer.write(example.SerializeToString())
    print('#%d Example Serialized String that written into stat.tfrecord:\n' % i, example.SerializeToString())
# Close writer
writer.close()
print('\n')


##############################   Read Data From TFRecords File  #############################################
# Create file name queue
filename_queue = tf.train.string_input_producer([data_path], num_epochs=2, shuffle=True)
# Create reader for reading TFRecords file
reader = tf.TFRecordReader()
# Read serialized data from stat.tfrecord
_, serialized_example = reader.read(filename_queue)
# Transfer serialized data into feature tensors
features = tf.parse_single_example(
    serialized_example,
    features={
        'id': tf.FixedLenFeature([], tf.int64),
        'age': tf.FixedLenFeature([], tf.int64),
        'income': tf.FixedLenFeature([], tf.float32),
        'outgo': tf.FixedLenFeature([], tf.float32)
    }
)
print('Features read from stat.tfrecord:\n', features)
print('\n')


################################  Use Data  ############################################
with tf.Session() as sess:
    # Init
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    # Create Coordinator
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Print background threads
    print('Threads: %s' % threads)

    try:
        for i in range(10):
            if not coord.should_stop():
                example = sess.run(features)
                print('#%d Read Data :\n' % i, example)

    except tf.errors.OutOfRangeError:
        print('Catch OutOfRangeError')

    finally:
        # Request to stop all background threads
        coord.request_stop()
        print('Finish Reading')

    # Wait for all background threads quit
    coord.join(threads)


