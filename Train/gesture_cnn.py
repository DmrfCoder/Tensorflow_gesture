# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from Utils.ReadAndDecode_Mic import read_and_decode

from Net.CNN_Init import weight_variable, bias_variable, conv2d, max_pool_2x2

log_path = '/home/dmrf/tensorflow_gesture_data/Log'
train_path = '/home/dmrf/tensorflow_gesture_data/Gesture_data/mic_train_5ms.tfrecords'
val_path = '/home/dmrf/tensorflow_gesture_data/Gesture_data/mic_test_5ms.tfrecords'
x_train, y_train = read_and_decode(train_path)
x_val, y_val = read_and_decode(val_path)

w = 550
h = 8
c = 2
labels_type = 13

# 占位符

# [batch, in_height, in_width, in_channels]
x = tf.placeholder(tf.float32, shape=[None, h, w, c], name='input')
y_label = tf.placeholder(tf.int64, shape=[None, ])

def add_net(in_x):
    # [filter_height, filter_width, in_channels, out_channels]
    w_conv1 = weight_variable([1, 7, 2, 16])
    b_conv1 = bias_variable([16])
    h_conv1 = tf.nn.relu(conv2d(in_x, w_conv1, [1, 1, 3,
                                                1]) + b_conv1)  # stride/kernel:The stride of the sliding window for each  dimension of `input`.

    h_pool1 = max_pool_2x2(h_conv1, [1, 1, 2, 1],
                           [1, 1, 2,
                            1])  # stride/kernel:The size of the window for each dimension of the input tensor.

    w_conv2 = weight_variable([1, 5, 16, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2, s=[1, 1, 2, 1]) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2, k=[1, 1, 2, 1], s=[1, 1, 2, 1])

    w_conv3 = weight_variable([1, 4, 32, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3, [1, 1, 2, 1]) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3, [1, 1, 2, 1], [1, 1, 2, 1])

    w_fc1 = weight_variable([8 * 6 * 64, 256])
    b_fc1 = bias_variable([256])
    h_pool3_flat = tf.reshape(h_pool3, [-1, 8 * 6 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1) + b_fc1,name="fullconnection1")

    w_fc2 = weight_variable([256, labels_type])
    b_fc2 = bias_variable([labels_type])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)

    out_y = tf.nn.softmax(h_fc2,name='softmax')
    return out_y

# Loss
y = add_net(x)
prediction_labels = tf.argmax(y, axis=1, name="output")
with tf.name_scope('loss'):
    base_lr = 0.5
    tf.summary.scalar('loss', base_lr) #写入tensorboard中的EVENTS

with tf.name_scope('loss'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_label, logits=y)
    tf.summary.scalar('loss', cross_entropy)

with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(learning_rate=base_lr).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), y_label)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 组合batch
train_batch = 64
test_batch = 32

min_after_dequeue_train = train_batch * 2
min_after_dequeue_test = test_batch * 2

num_threads = 3 #开启3个线程

train_capacity = min_after_dequeue_train + num_threads * train_batch
test_capacity = min_after_dequeue_test + num_threads * test_batch

Training_iterations = 4500 #训练迭代次数
Validation_size = 100   #每隔100次在屏幕上打印一次
test_count = labels_type * 100 #测试样本总数
Test_iterations = test_count / test_batch #测试迭代次数

# 使用shuffle_batch可以随机打乱输入
train_x_batch, train_y_batch = tf.train.shuffle_batch([x_train, y_train],
                                                      batch_size=train_batch, capacity=train_capacity,
                                                      min_after_dequeue=min_after_dequeue_train)

# 使用shuffle_batch可以随机打乱输入
test_x_batch, test_y_batch = tf.train.shuffle_batch([x_val, y_val],
                                                    batch_size=test_batch, capacity=test_capacity,
                                                    min_after_dequeue=min_after_dequeue_test)

with tf.Session() as sess:
    merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter("../Logs/", sess.graph)
    sess.run(tf.global_variables_initializer())
    threads = tf.train.start_queue_runners(sess=sess)
    for step in range(Training_iterations + 1):
        train_x, train_y = sess.run([train_x_batch, train_y_batch])

        sess.run(train, feed_dict={x: train_x, y_label: train_y})
        # Train accuracy
        if step % Validation_size == 0:
            # base_lr = adjust_learning_rate_inv(step, base_lr)
            a = sess.run(accuracy, feed_dict={x: train_x, y_label: train_y})
            print('Training Accuracy', step, a)

            #    plot_confusion_matrix(correct_prediction,y_label)
            result = sess.run(merged,
                              feed_dict={x: train_x, y_label: train_y})
            writer.add_summary(result, step)
            # constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
            # with tf.gfile.FastGFile('gesture_cnn.pb', mode='wb') as f:
            #     f.write(constant_graph.SerializeToString())

    for step in range(Test_iterations + 1):
        test_x, test_y = sess.run([test_x_batch, test_y_batch])
        b = sess.run(accuracy, feed_dict={x: test_x, y_label: test_y})
        print('Test Accuracy', step,b)

    # output_graph_def = tf.graph_until.convert_variables_to_constants(sess, sess.graph_def,
    #                                                                  output_node_names=['output'])
    # with tf.gfile.FastGFile('gesture_cnn.pb', mode = 'wb') as f:
    #     f.write(output_graph_def.SerializeToString())
    constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])

    with tf.gfile.FastGFile('../Model/gesture_cnn256.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())