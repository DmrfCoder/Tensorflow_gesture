# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

SUMMART_DIR="../Logs/mnistLogs/" #log路径
BATCH_SIZE = 100    #一次批处理大小
TRAIN_STEPS = 30000 #训练迭代次数
#定义生成监控信息日志的操作 var:张量 name:变量名
def variable_summaries(oneTensor,name):
    with tf.name_scope('summaries'):
        tf.summary.histogram(name,oneTensor) #该函数记录var中的元素的取值分布
        mean=tf.reduce_mean(oneTensor)   #计算变量平均值
        tf.summary.scalar('mean/'+name,mean) #添加到日志中
        stddev=tf.sqrt(tf.reduce_mean(tf.square(oneTensor-mean)))#计算变量标准差
        tf.summary.scalar('stddev/'+name,stddev)#添加到日志中
#定义全连接层 input_dim:输入维度,output_dim:输出维度 默认为relu激活函数
def nn_layer(input_tensor,input_dim,output_dim,
             layer_name,act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            #声明权重 tf.truncated_normal生成一份符合正态分布的数据
            weights=tf.Variable(tf.truncated_normal(shape=[input_dim,output_dim],stddev=0.1))
            variable_summaries(weights,layer_name+'/weights')
        with tf.name_scope('biases'):
            #声明偏置项
            biases = tf.Variable(tf.constant(0.0,shape=[output_dim]))
            variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            preactivate=tf.matmul(input_tensor,weights)+biases
            tf.summary.histogram(layer_name+'/pre_actuvations',preactivate)
        activations=act(preactivate,name='activation')
        tf.summary.histogram(layer_name+'/activations',activations)
        return activations
# def main(_):
mnist=input_data.read_data_sets("../Demo/MNIST_data/",one_hot=True)
with tf.name_scope('input'):
    #定义输入,声明占位符
    x=tf.placeholder(tf.float32,[None,784],name='x-input')
    y_=tf.placeholder(tf.float32, [None, 10], name='y-input')
#生成图片的操作
# with tf.name_scope('input_reshape'):
#     image_shaped_input=tf.reshape(x,[-1,28,28,1])
#     tf.summary.image('input',image_shaped_input,10)
hidden1=nn_layer(x,784,500,'layer1')
y=nn_layer(hidden1,500,10,'layer2',act=tf.identity)
with tf.name_scope('cross_entropy'):
    #计算交叉熵
    cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_))
    tf.summary.scalar('cross entroy',cross_entropy)
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    with tf.name_scope('accuracy'):
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar('accuracy',accuracy)
merged=tf.summary.merge_all()
with tf.Session() as sess:
    summary_writer= tf.summary.FileWriter(SUMMART_DIR,sess.graph)
    tf.initialize_all_variables().run()
    for i in range(TRAIN_STEPS):
        xs,ys=mnist.train.next_batch(BATCH_SIZE)
        summary,_=sess.run([merged,train_step],feed_dict={x:xs,y_:ys})
        summary_writer.add_summary(summary,i)
summary_writer.close()





















