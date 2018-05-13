# -*-coding:utf-8-*-
from __future__ import absolute_import, unicode_literals
import tensorflow as tf
import numpy as np
import os

test_batch=64
pb_file_path = "../Model/gesture_cnn256addlstm.pb"
pb_lstm_file_path = "../Model/gesture_lstm.pb"
output_graph_def = tf.GraphDef()
with open(pb_file_path, "rb") as f:
    output_graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(output_graph_def, name="")

def testdemo():
    with tf.Session() as sess:
        init = tf.global_variables_initializer()

        sess.run(init)



        input_x_lstm = sess.graph.get_tensor_by_name("input_lstm:0")
        print input_x_lstm
        softmax_lstm = sess.graph.get_tensor_by_name("softmax_lstm:0")
        print softmax_lstm
        x_ndarry_lstm = np.zeros(shape=(test_batch, 1024), dtype=np.float32)
        out_softmax = sess.run(softmax_lstm, feed_dict={input_x_lstm: x_ndarry_lstm})
        print(out_softmax)

if __name__=='__main__':

    testdemo()
