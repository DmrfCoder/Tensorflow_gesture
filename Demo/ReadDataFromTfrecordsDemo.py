# -*-coding:utf-8-*-
import tensorflow as tf

from Utils.ReadAndDecode import read_and_decode

if __name__ == "__main__":
    img, label = read_and_decode("/home/dmrf/tensorflow_gesture_data/Gesture_data/abc_mic_train_5.tfrecords")

    # 使用shuffle_batch可以随机打乱输入
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=30, capacity=2000,
                                                    min_after_dequeue=1000)
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess)
        for i in range(3):
            val, l = sess.run([img_batch, label_batch])
            # 我们也可以根据需要对val， l进行处理
            # l = to_categorical(l, 12)
            print(val.shape, l)
