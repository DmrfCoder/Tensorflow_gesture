# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import rnn
from Utils.ReadAndDecode_Mic import read_and_decode

log_path = '/home/wjyyy/Tensorflow/Log'
train_path = '/home/wjyyy/Tensorflow/Data/mic_train_5ms.tfrecords'
val_path = '/home/wjyyy/Tensorflow/Data/mic_test_5ms.tfrecords'
pb_file_path = "../Model/gesture_cnn.pb"

output_graph_def = tf.GraphDef()

with open(pb_file_path, "rb") as f:
    output_graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(output_graph_def, name="")

# log_path = '/home/wjyyy/Tensorflow/Log'
# train_path = '/home/wjyyy/Tensorflow/Data/abc_mic_train_5.tfrecords'
# val_path = '/home/wjyyy/Tensorflow/Data/abc_mic_val_5.tfrecords'
x_train, y_train = read_and_decode(train_path)
x_val, y_val = read_and_decode(val_path)

n_steps = 4  # time steps
n_inputs = 256
n_classes = 8
n_hidden_units = 256  # neurons in hidden layer
n_layer_num = 2

# 占位符
# RNN 的输入shape = (batch_size, timestep_size, input_size)
x_lstm = tf.placeholder(tf.float32, shape=[None, n_steps*n_inputs])#4*256 vector
y_label = tf.placeholder(tf.int64, shape=[None, ])
keep_prob = tf.placeholder(tf.float32)


weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

batch_size = 1


def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tenosors list of shape (batch_size, n_input)
    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.reshape(x, [-1, n_steps, n_inputs])

    lstm_cell = rnn.BasicLSTMCell(num_units=n_hidden_units, forget_bias=1.0, state_is_tuple=True)

    lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)

    mlstm_cell = rnn.MultiRNNCell([lstm_cell] * n_layer_num, state_is_tuple=True)

    init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(mlstm_cell, inputs=x, initial_state=init_state, time_major=False)
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # 选取最后一个 output

    return results


# Loss
logist = RNN(x_lstm, weights, biases)
logist = tf.nn.softmax(logist)
prediction_labels = tf.argmax(logist, axis=1)

base_lr = 0.5

cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_label, logits=logist)

Optimizer = tf.train.GradientDescentOptimizer(learning_rate=base_lr)
train = Optimizer.minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(prediction_labels, 1), y_label)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 组合batch
train_batch = batch_size
test_batch = batch_size

min_after_dequeue_train = train_batch * 2
min_after_dequeue_test = test_batch * 2

num_threads = 3

train_capacity = min_after_dequeue_train + num_threads * train_batch
test_capacity = min_after_dequeue_test + num_threads * test_batch

Training_iterations = 3000
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
    input_x = sess.graph.get_tensor_by_name("input:0")
    print input_x
    out_softmax = sess.graph.get_tensor_by_name("softmax:0")
    print out_softmax
    fc = sess.graph.get_tensor_by_name("fullconnection1:0")
    print fc

    for step in range(Training_iterations + 1):
        train_x, train_y = sess.run([train_x_batch, train_y_batch])

        x_lstm= tf.zeros(shape=[None, n_steps*n_inputs],dtype=tf.float32)
        x__list_lstm = [0] * 1024  # 定义一个长度为1024的list
        x_list_cnn=[0]*8800

        #tfrecords-->tensor(8,2200,2)-->4*tensor(8,550,2)-->cnn-->4*256-->lstm



        if train_x[0][1][1100][0] == 1 * 6:  # 0.5s


        else if train_x[0][1][1100][0] == 2 * 6:  # 1s

        else:  # 2s  else is needed

        out_fc = sess.run(fc, feed_dict={input_x: train_x})

        sess.run(train, feed_dict={x: out_fc, y_label: train_y})
        # Train accuracy
        if step % Validation_size == 0:
            # base_lr = adjust_learning_rate_inv(step, base_lr)
            a = sess.run(accuracy, feed_dict={x: train_x, y_label: train_y})
            print('Training Accuracy', step, a
                  )

    for step in range(Test_iterations + 1):
        test_x, test_y = sess.run([test_x_batch, test_y_batch])
        x_lstm = tf.zeros(shape=[None, n_steps * n_inputs], dtype=tf.float32)
        out_fc = sess.run(fc, feed_dict={input_x: test_x})
        b = sess.run(accuracy, feed_dict={x: out_fc, y_label: train_y})
        print('Test Accuracy', step,
              b)
