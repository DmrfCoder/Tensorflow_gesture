# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import rnn
from Utils.ReadAndDecode import read_and_decode

from Net.CNN_Init import weight_variable, bias_variable, conv2d, max_pool_2x2
log_path = '/home/wjyyy/Tensorflow/Log'
train_path = '/home/wjyyy/Tensorflow/Data/mic_train_5ms.tfrecords'
val_path = '/home/wjyyy/Tensorflow/Data/mic_test_5ms.tfrecords'
#log_path = '/home/wjyyy/Tensorflow/Log'
#train_path = '/home/wjyyy/Tensorflow/Data/abc_mic_train_5.tfrecords'
#val_path = '/home/wjyyy/Tensorflow/Data/abc_mic_val_5.tfrecords'
x_train, y_train = read_and_decode(train_path)
x_val, y_val = read_and_decode(val_path)

n_steps = 8  # time steps
n_inputs = 550
c = 2
n_classes = 13
n_hidden_units = 128  # neurons in hidden layer

# 占位符

# [batch, in_height, in_width, in_channels]

x = tf.placeholder(tf.float32, shape=[None, n_steps, n_inputs, c])
y_label = tf.placeholder(tf.int64, shape=[None, ])


weights = {
    'in': tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1,shape =[n_hidden_units,])),
    'out': tf.Variable(tf.constant(0.1,shape =[n_classes,]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tenosors list of shape (batch_size, n_input)
    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.reshape(x,[-1,n_inputs])
    x_in = tf.matmul(x, weights['in']) + biases['in']
    x_in = tf.reshape(x_in,[-1, n_steps, n_hidden_units])
    x_in = tf.unstack(x_in, n_steps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x_in, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']



# Loss
logits = RNN(x, weights, biases)
logits = tf.reshape(logits, [64,-1])
prediction = tf.nn.softmax(logits)

base_lr = 0.5

cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_label, logits=prediction)
Optimizer = tf.train.GradientDescentOptimizer(learning_rate=base_lr)
train = Optimizer.minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(prediction, 1), y_label)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 组合batch
train_batch = 64
test_batch = 32

min_after_dequeue_train = train_batch * 2
min_after_dequeue_test = test_batch * 2

num_threads = 3

train_capacity = min_after_dequeue_train + num_threads * train_batch
test_capacity = min_after_dequeue_test + num_threads * test_batch

Training_iterations = 15000
Validation_size = 100

test_count = n_classes * 100
Test_iterations = test_count / test_batch

display_step = 100

# 使用shuffle_batch可以随机打乱输入
train_x_batch, train_y_batch = tf.train.shuffle_batch([x_train, y_train],
                                                      batch_size=train_batch, capacity=train_capacity,
                                                      min_after_dequeue=min_after_dequeue_train)

# 使用shuffle_batch可以随机打乱输入
test_x_batch, test_y_batch = tf.train.shuffle_batch([x_val, y_val],
                                                    batch_size=test_batch, capacity=test_capacity,
                                                    min_after_dequeue=min_after_dequeue_test)

# Train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    threads = tf.train.start_queue_runners(sess=sess)
    for step in range(Training_iterations + 1):
        train_x, train_y = sess.run([train_x_batch, train_y_batch])

        sess.run(train, feed_dict={x: train_x, y_label: train_y})
        # Train accuracy
        if step % Validation_size == 0:
            # base_lr = adjust_learning_rate_inv(step, base_lr)
            a = sess.run(accuracy, feed_dict={x: train_x, y_label: train_y})
            print('Training Accuracy', step, a
                  )

    for step in range(Test_iterations + 1):
        test_x, test_y = sess.run([test_x_batch, test_y_batch])
        b = sess.run(accuracy, feed_dict={x: train_x, y_label: train_y})
        print('Test Accuracy', step,
              b)
