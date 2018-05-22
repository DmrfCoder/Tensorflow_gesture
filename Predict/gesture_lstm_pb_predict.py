# -*-coding:utf-8-*-
from __future__ import absolute_import, unicode_literals
import tensorflow as tf
import numpy as np
import os

import matplotlib.pyplot as plt

from Utils.ReadAndDecode_Continous import read_and_decode_continous

val_path = '/home/dmrf/GestureNuaaTeam/tensorflow_gesture_data/Gesture_data/continous_data/test_continous.tfrecords'
# val_path = '/home/dmrf/GestureNuaaTeam/tensorflow_gesture_data/Gesture_data/continous_data/train_continous.tfrecords'
#val_path = '/home/dmrf/GestureNuaaTeam/tensorflow_gesture_data/Gesture_data/abij_test.tfrecords'
# val_path = '/home/dmrf/GestureNuaaTeam/tensorflow_gesture_data/Gesture_data/abij_train.tfrecords'
x_val, y_val = read_and_decode_continous(val_path)

test_batch = 1
min_after_dequeue_test = test_batch * 2

num_threads = 3
test_capacity = min_after_dequeue_test + num_threads * test_batch

# 使用shuffle_batch可以随机打乱输入
test_x_batch, test_y_batch = tf.train.shuffle_batch([x_val, y_val],
                                                    batch_size=test_batch, capacity=test_capacity,
                                                    min_after_dequeue=min_after_dequeue_test)

labels_type = 8
test_count = labels_type * 100
Test_iterations = test_count / test_batch

output_graph_def = tf.GraphDef()

pb_file_path = "../Model/gesture_cnn_lstm.pb"
pb_lstm_file_path = "../Model/gesture_lstm.pb"

with open(pb_file_path, "rb") as f:
    output_graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(output_graph_def, name="")

# with open(pb_lstm_file_path, "rb") as f:
#     output_graph_def.ParseFromString(f.read())
#     _ = tf.import_graph_def(output_graph_def, name="")

# LABELS = ['A', 'B', 'C', 'F', 'G', 'H', 'I', 'J']
# LABELS = ['A', 'B',  'I', 'J']
label = [0, 1, 2, 3, 4, 5, 6, 7]


def batchtest():
    re_label = np.ndarray(803, dtype=np.int64)
    pr_label = np.ndarray(803, dtype=np.int64)
    ind=0
    with tf.Session() as sess:
        init = tf.global_variables_initializer()

        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess)

        input_x = sess.graph.get_tensor_by_name("input:0")
        print input_x
        fc = sess.graph.get_tensor_by_name("fullconnection1:0")
        print fc

        output_cnn = sess.graph.get_tensor_by_name("output:0")
        print output_cnn

        input_x_lstm = sess.graph.get_tensor_by_name("input_lstm:0")
        print input_x_lstm

        softmax_lstm = sess.graph.get_tensor_by_name("softmax_lstm:0")
        print softmax_lstm
        output_lstm = sess.graph.get_tensor_by_name("output_lstm:0")
        print output_lstm

        for step_test in range(Test_iterations + 1):
            test_x, test_y = sess.run([test_x_batch, test_y_batch])

            x_ndarry_lstm = np.zeros(shape=(test_batch, 1024), dtype=np.float32)  # 定义一个长度为1024的array
            # if test_y==1:
            #     x_ndarry_lstm[:, 0:256] = sess.run(fc, feed_dict={input_x: test_x[:, :, 0:550]})
            #     out= sess.run(output_cnn, feed_dict={input_x: test_x[:, :, 0:550]})
            #     out_lstm=sess.run(output_lstm, feed_dict={input_x_lstm: x_ndarry_lstm})
            #     print out,out_lstm

            # tfrecords-->tensor(8,2200,2)-->4*tensor(8,550,2)-->cnn-->4*256-->lstm

            # train_x[0][1][1100][0] is the flag when write tfrecord
            if test_x[0][1][1100][0] == 1 * 6:  # 0.5s-->need train_x[:][1][0:550][0]

                x_ndarry_lstm[:, 0:256] = sess.run(fc, feed_dict={input_x: test_x[:, :, 0:550]})

            elif test_x[0][1][1100][0] == 2 * 6:  # 1s-->need train_x[:][1][0:1100][0]
                x_ndarry_lstm[:, 0:256] = sess.run(fc, feed_dict={input_x: test_x[:, :, 0:550]})
                x_ndarry_lstm[:, 256:512] = sess.run(fc, feed_dict={input_x: test_x[:, :, 550:1100]})

                # x_narray_cnn[0][:] = train_x[:][:][0:550]
                # x_narray_cnn[1][:] = train_x[:][:][550:1100]

            else:  # 2s-->need train_x[:][1][0:2200][0]
                x_ndarry_lstm[:, 0:256] = sess.run(fc, feed_dict={input_x: test_x[:, :, 0:550]})
                x_ndarry_lstm[:, 256:512] = sess.run(fc, feed_dict={input_x: test_x[:, :, 550:1100]})
                x_ndarry_lstm[:, 512:768] = sess.run(fc, feed_dict={input_x: test_x[:, :, 1100:1650]})
                x_ndarry_lstm[:, 768:1024] = sess.run(fc, feed_dict={input_x: test_x[:, :, 1650:2200]})
                # x_narray_cnn[0][:] = train_x[:][:][0:550]
                # x_narray_cnn[1][:] = train_x[:][:][550:1100]
                # x_narray_cnn[2][:] = train_x[:][:][1100:1650]
                # x_narray_cnn[3][:] = train_x[:][:][1650:2200]

            # print 0
            prediction_labels = sess.run(output_lstm, feed_dict={input_x_lstm: x_ndarry_lstm})
            print(str(step_test))
            print "real_label:", test_y


            re_label[ind] = test_y

            # prediction_labels = np.argmax(out_softmax, axis=1)
            pr_label[ind] = prediction_labels
            ind+=1
            print "predict_label:", prediction_labels
            print('')

    np.savetxt('../Data/re_label_lstmtrain_addlstm.txt', re_label)
    np.savetxt('../Data/pr_label_lstmtrain_addlstm.txt', pr_label)


if __name__ == '__main__':
    # singletest_data_pc("/home/dmrf/test_gesture/JS")
    # ReadDataFromTxt("/home/dmrf/下载/demodata/0_push left_1524492872166_1")
    batchtest()
