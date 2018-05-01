# -*-coding:utf-8-*-
from __future__ import absolute_import, unicode_literals
import tensorflow as tf
import numpy as np

from Utils.ReadAndDecode import read_and_decode

val_path = '/home/caiyiwu/下载/data18.4.22/mic_test_5ms.tfrecords'
x_val, y_val = read_and_decode(val_path)

test_batch = 1
min_after_dequeue_test = test_batch * 2

num_threads = 3
test_capacity = min_after_dequeue_test + num_threads * test_batch

# 使用shuffle_batch可以随机打乱输入
test_x_batch, test_y_batch = tf.train.shuffle_batch([x_val, y_val],
                                                    batch_size=test_batch, capacity=test_capacity,
                                                    min_after_dequeue=min_after_dequeue_test)

labels_type = 13
test_count = labels_type * 100
Test_iterations = test_count / test_batch

output_graph_def = tf.GraphDef()


pb_file_path="../Model/gesture_cnn.pb"

re_label= np.ndarray(1301, dtype=np.int64)
pr_label= np.ndarray(1301, dtype=np.int64)


with open(pb_file_path, "rb") as f:
    output_graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(output_graph_def, name="")

with tf.Session() as sess:
    init = tf.global_variables_initializer()

    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)

    input_x = sess.graph.get_tensor_by_name("input:0")
    print input_x
    out_softmax = sess.graph.get_tensor_by_name("softmax:0")
    print out_softmax
    # out_label = sess.graph.get_tensor_by_name("output:0")
    # print out_label

    for step in range(Test_iterations + 1):
        test_x, test_y = sess.run([test_x_batch, test_y_batch])

        img_out_softmax = sess.run(out_softmax, feed_dict={input_x: test_x})

        print(str(step))
        print "real_label:", test_y
        re_label[step]=test_y
        prediction_labels = np.argmax(img_out_softmax, axis=1)
        pr_label[step]=prediction_labels
        print "predict_label:", prediction_labels
        print('')


np.savetxt('../Data/re_label.txt',re_label)
np.savetxt('../Data/pr_label.txt',pr_label)

