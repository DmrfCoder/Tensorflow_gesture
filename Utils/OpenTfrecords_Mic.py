# -*-coding:utf-8-*-
# 读取文件。
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(
    ["/home/dmrf/tensorflow_gesture_data/Gesture_data2/abc_mic_train_5.tfrecords"])
_, serialized_example = reader.read(filename_queue)

# 解析读取的样例。
features = tf.parse_single_example(
    serialized_example,
    features={
        'label': tf.FixedLenFeature([], tf.int64),
        'data_raw': tf.FixedLenFeature([], tf.string)
    })

images = tf.decode_raw(features['data_raw'], tf.float64)
images = tf.reshape(images, [8, 550, 2])
# images = tf.cast(images, tf.float32) * (1. / 255) - 0.5  # 将图片中的数据转为[-0.5,0.5]

#labels = tf.decode_raw(features['label'], tf.int64)
labels = tf.cast(features['label'], tf.int64)



sess = tf.Session()

# 启动多线程处理输入数据。
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for i in range(3):
    image, label = sess.run([images, labels])

    print label, image
    # plt.plot(image)
    # plt.show()
    # break
