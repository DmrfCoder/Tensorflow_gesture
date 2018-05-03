# -*-coding:utf-8-*-
from __future__ import absolute_import, unicode_literals
import tensorflow as tf
import numpy as np
import os
from Utils.ReadAndDecode_Mic import read_and_decode_mic
import matplotlib.pyplot as plt

val_path = '/home/dmrf/tensorflow_gesture_data/Gesture_data/mic_test_5ms.tfrecords'
x_val, y_val = read_and_decode_mic(val_path)

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

pb_file_path = "../Model/gesture_cnn256.pb"

with open(pb_file_path, "rb") as f:
    output_graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(output_graph_def, name="")

LABELS = ['A', 'B', 'C', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']


def ReadDataFromTxt(path):
    files2 = os.listdir(path)
    if "I" in files2[0]:
        I = np.loadtxt(path + '/' + files2[0])
        Q = np.loadtxt(path + '/' + files2[1])
    else:
        try:
            I = np.loadtxt(path + '/' + files2[1])
            Q = np.loadtxt(path + '/' + files2[0])
        except ValueError:
            return "error"

    data = np.ndarray((1, 8, 550, 2), dtype=np.float64)
    if len(I) != 4400:
        print 'error'

    if len(Q) != 4400:
        print 'error'

    I = I.reshape(8, 550)
    Q = Q.reshape(8, 550)

    for i in range(0, 8):
        for j in range(0, 550):
            data[0][i][j][0] = I[i][j]
            data[0][i][j][1] = Q[i][j]

    return data


def batchtest():
    re_label = np.ndarray(1301, dtype=np.int64)
    pr_label = np.ndarray(1301, dtype=np.int64)
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
            # x shape 8*550*2
            test_x, test_y = sess.run([test_x_batch, test_y_batch])

            img_out_softmax = sess.run(out_softmax, feed_dict={input_x: test_x})

            print(str(step))
            print "real_label:", test_y
            re_label[step] = test_y
            prediction_labels = np.argmax(img_out_softmax, axis=1)
            pr_label[step] = prediction_labels
            print "predict_label:", prediction_labels
            print('')

    np.savetxt('../Data/re_label.txt', re_label)
    np.savetxt('../Data/pr_label.txt', pr_label)


def singletest_data_ad(path):
    files = os.listdir(path)
    l = len(files)
    l = l + 1
    si_re_label = np.ndarray(l, dtype=np.int64)
    si_pr_label_tf = np.ndarray(l, dtype=np.int64)
    si_pr_label_ad = np.ndarray(l, dtype=np.int64)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()

        sess.run(init)

        input_x = sess.graph.get_tensor_by_name("input:0")
        print input_x
        out_softmax = sess.graph.get_tensor_by_name("softmax:0")
        print out_softmax
        index = 0
        for file in files:
            type = -1
            if "click" in file:
                if file[-1] == '0':
                    type = 7
                if file[-1] == '1':
                    type = 8
            if "flip" in file:
                if file[-1] == '0':
                    type = 3
                if file[-1] == '1':
                    type = 4
            if "grab" in file:
                type = 5
            if "left" in file:
                type = 1
            if "right" in file:
                type = 2
            if "static" in file:
                type = 0

            test_x = ReadDataFromTxt(path + '/' + file)

            img_out_softmax = sess.run(out_softmax, feed_dict={input_x: test_x})

            prediction_labels = np.argmax(img_out_softmax, axis=1)
            print "tf_predict_label:", prediction_labels, "android_label:", file[0], "really_label", type
            si_re_label[index] = type
            si_pr_label_ad[index] = file[0]
            si_pr_label_tf[index] = prediction_labels
            index = index + 1

        np.savetxt('../Data/si_re_label.txt', si_re_label)
        np.savetxt('../Data/si_pr_ad.txt', si_pr_label_ad)
        np.savetxt('../Data/si_pr_tf.txt', si_pr_label_tf)


def singletest_data_pc(path):
    files = os.listdir(path)
    l = len(files)
    pc_re_label = np.zeros(2 * 1001, dtype=np.int64)
    pc_pr_label_tf = np.zeros(2 * 1001, dtype=np.int64)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        input_x = sess.graph.get_tensor_by_name("input:0")
        print input_x
        out_softmax = sess.graph.get_tensor_by_name("softmax:0")
        print out_softmax
        index = 0
        files.sort()
        for file in files:
            test_x = ReadDataFromTxt(path + '/' + file)

            if test_x == "error":
                continue

            img_out_softmax = sess.run(out_softmax, feed_dict={input_x: test_x})

            prediction_labels = np.argmax(img_out_softmax, axis=1)
            print "filename:", file, " tf_predict_label:", prediction_labels
            pc_pr_label_tf[index] = prediction_labels
            pc_re_label[index] = LABELS.index(file[0])
            index = index + 1
            if index == 2200:
                break
        np.savetxt('../Data/pc_re_label.txt', pc_re_label)
        np.savetxt('../Data/pc_pr_label_tf.txt', pc_pr_label_tf)


def gg(path):
    files = os.listdir(path)
    for file in files:
        if file[16] == 'A':
            os.rename(path + "/" + file, path + "/" + file[16:])
        elif file[16] == 'B':
            os.rename(path + "/" + file, path + "/" + file[16:])
        elif file[16] == 'C':
            os.rename(path + "/" + file, path + "/" + file[16:])
        elif file[16] == 'I':
            os.rename(path + "/" + file, path + "/H" + file[17:])
        elif file[16] == 'J':
            os.rename(path + "/" + file, path + "/I" + file[17:])
        elif file[16] == 'F':
            if file[-2:] == '_2':
                os.rename(path + "/" + file, path + "/K" + file[17:])
            else:
                os.rename(path + "/" + file, path + "/J" + file[17:])
        elif file[16] == 'G':
            if file[-2:] == '_2':
                os.rename(path + "/" + file, path + "/G" + file[17:])
            else:
                os.rename(path + "/" + file, path + "/F" + file[17:])


if __name__ == '__main__':
    singletest_data_pc("/home/dmrf/下载/demodata")
    #gg("/home/dmrf/下载/demodata")
    # ReadDataFromTxt("/home/dmrf/下载/demodata/lizhenyan_04_27_A_1524795830456_2")
