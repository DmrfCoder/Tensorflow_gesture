# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from Utils.ReadAndDecode import read_and_decode
from Net.CNN_Init import weight_variable, bias_variable, conv2d, max_pool_2x2

log_path='../Logs/gestureLogs/'
train_path = '/home/caiyiwu/下载/data18.4.22/mic_train_5ms.tfrecords'
val_path = '/home/caiyiwu/下载/data18.4.22/mic_test_5ms.tfrecords'
x_train, y_train = read_and_decode(train_path)
x_val, y_val = read_and_decode(val_path)

#全局变量
w = 550 #0.5秒的数据量
h = 8 #八个频率
c = 2 #通道数
labels_type = 13#微手势种类数
train_batch = 64 #训练时批处理大小
test_batch = 32 #测试时批处理大小
min_after_dequeue_train = train_batch * 2 #队列最小训练大小
min_after_dequeue_test = test_batch * 2 #队列最小测试大小
num_threads = 3 #开启3个线程
train_capacity = min_after_dequeue_train + num_threads * train_batch #5*train_batch
test_capacity = min_after_dequeue_test + num_threads * test_batch    #5*test_batch
Training_iterations = 4500 #训练迭代次数
Validation_size = 100   #每隔100次在屏幕上打印一次
test_count = labels_type * 100 #测试样本总数
Test_iterations = test_count / test_batch #测试迭代次数
base_lr = 0.05 #学习率

def variable_summaries(oneTensor,name):
    tf.summary.histogram(name,oneTensor) #该函数记录var中的元素的取值分布
    mean=tf.reduce_mean(oneTensor)   #计算变量平均值
    tf.summary.scalar('average',mean) #添加到日志中
    stddev=tf.sqrt(tf.reduce_mean(tf.square(oneTensor-mean)))#计算变量标准差
    tf.summary.scalar('stddev',stddev)#添加到日志中

def add_net(in_x):
    with tf.name_scope('CNN-Net'):
        # [filter_height, filter_width, in_channels, out_channels]
        with tf.name_scope("weight_conv1"):
            w_conv1 = weight_variable([1, 7, 2, 16])
            variable_summaries(w_conv1,"weight_conv1")
        with tf.name_scope("biases_conv1"):
            b_conv1 = bias_variable([16])
            variable_summaries(b_conv1,"biases_conv1")
        h_conv1 = tf.nn.relu(conv2d(in_x, w_conv1, [1, 1, 3,1]) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1, [1, 1, 2, 1],[1, 1, 2,1])

        with tf.name_scope("weight_conv2"):
            w_conv2 = weight_variable([1, 5, 16, 32])
            variable_summaries(w_conv2,"weight_conv2")
        with tf.name_scope("biases_conv2"):
            b_conv2 = bias_variable([32])
            variable_summaries(b_conv2,"biases_conv2")
        h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2, s=[1, 1, 2, 1]) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2, k=[1, 1, 2, 1], s=[1, 1, 2, 1])

        with tf.name_scope("weight_conv3"):
            w_conv3 = weight_variable([1, 4, 32, 64])
            variable_summaries(w_conv3,"weight_conv3")
        with tf.name_scope("biases_conv3"):
            b_conv3 = bias_variable([64])
            variable_summaries(b_conv3,"biases_conv3")
        h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3, [1, 1, 2, 1]) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3, [1, 1, 2, 1], [1, 1, 2, 1])

        with tf.name_scope("w_fc1"):
            w_fc1 = weight_variable([8 * 6 * 64, 256])
            variable_summaries(w_fc1,"w_fc1")
        with tf.name_scope("b_fc1"):
            b_fc1 = bias_variable([256])
            variable_summaries(b_fc1,"b_fc1")
        h_pool3_flat = tf.reshape(h_pool3, [-1, 8 * 6 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1) + b_fc1)

        with tf.name_scope("w_fc2"):
            w_fc2 = weight_variable([256, labels_type])
            variable_summaries(w_fc2,"w_fc2")
        with tf.name_scope("b_fc2"):
            b_fc2 = bias_variable([labels_type])
            variable_summaries(b_fc2,"b_fc2")
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)

        with tf.name_scope("softmax"):
            out_y = tf.nn.softmax(h_fc2)
            variable_summaries(out_y, "softmax")
        return out_y

with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, shape=[None, h, w, c], name='x-input')
    y_label = tf.placeholder(tf.int64, shape=[None,],name='y-input')

y = add_net(x)
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=y,labels=y_label)
    tf.summary.scalar('cross_entropy', cross_entropy)
with tf.name_scope('train'):
    with tf.name_scope('learning_rate'):
        learning_rate=tf.train.exponential_decay(base_lr,tf.Variable(0),Validation_size,0.96,staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)
    with tf.name_scope('train'):
        #待定 不知道需不需要保存
        train= tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
        # tf.summary.scalar('train',train)
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        #用tf.argmax(y_label,1)会出错
        correct_prediction = tf.equal(y_label,tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar('accuracy',accuracy)

# 使用shuffle_batch可以随机打乱输入
train_x_batch, train_y_batch = tf.train.shuffle_batch([x_train, y_train],
                                                      batch_size=train_batch, capacity=train_capacity,
                                                      min_after_dequeue=min_after_dequeue_train)
test_x_batch, test_y_batch = tf.train.shuffle_batch([x_val, y_val],
                                                    batch_size=test_batch, capacity=test_capacity,
                                                    min_after_dequeue=min_after_dequeue_test)

merged = tf.summary.merge_all()
with tf.Session() as sess:
    writer = tf.summary.FileWriter(log_path, sess.graph)
    sess.run(tf.global_variables_initializer())
    #这一步好像没用到?但是没有的话无法更新train_x, train_y
    threads = tf.train.start_queue_runners(sess=sess)
    for step in range(Training_iterations):
        train_x, train_y = sess.run([train_x_batch, train_y_batch])
        sess.run(train, feed_dict={x: train_x, y_label: train_y})
        #写入日志
        summary,_=sess.run([merged, train], feed_dict={x: train_x, y_label: train_y})
        writer.add_summary(summary, step)
        if step % Validation_size == 0:
            a=sess.run(accuracy, feed_dict={x: train_x, y_label: train_y})
            print('Training Accuracy', step, a)

    for step in range(Test_iterations + 1):
        test_x, test_y = sess.run([test_x_batch, test_y_batch])
        b = sess.run(accuracy, feed_dict={x: test_x, y_label: test_y})
        print('Test Accuracy', step,b)
    writer.close()
    #输出成.pb文件供画图使用
    constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
    with tf.gfile.FastGFile('../Model/gesture_cnn.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())