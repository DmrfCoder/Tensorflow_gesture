# -*-coding:utf-8-*-
from __future__ import absolute_import, unicode_literals
import tensorflow as tf
import numpy as np
import os
from Utils.ReadAndDecode_Mic import read_and_decode
import matplotlib.pyplot as plt

output_graph_def = tf.GraphDef()

pb_file_path = "../Model/gesture_cnn256.pb"

with open(pb_file_path, "rb") as f:
    output_graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(output_graph_def, name="")


def FixWeightCnn():
    with tf.Session() as sess:
        init = tf.global_variables_initializer()

        sess.run(init)

        input_x = sess.graph.get_tensor_by_name("input:0")
        print input_x
        out_softmax = sess.graph.get_tensor_by_name("softmax:0")
        print out_softmax
        fc = sess.graph.get_tensor_by_name("fullconnection1:0")
        print fc


if __name__ == '__main__':
    FixWeightCnn()
