# -*- coding:utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 导入测试数据
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_data", one_hot=True)

# 初始化变量
input_x = tf.placeholder(tf.float32, [None, 28 * 28]) / 255.0
output_y = tf.placeholder(tf.int32, [None, 10])

# 改变形状的input_x
input_x_images = tf.reshape(input_x, [-1, 28, 28, 1])

# 选取3000个数据集
test_x = mnist.train.images[:3000]
test_y = mnist.train.labels[:3000]

conv1 = tf.layers.conv2d(
    inputs=input_x_images,
    kernel_size=[5, 5],
    filters=32,
    strides=1,
    padding="same",
    activation=tf.nn.relu
)

pool1 = tf.layers.max_pooling2d(
    inputs=conv1,
    strides=2,
    pool_size=[2, 2]
)

conv2 = tf.layers.conv2d(
    inputs=pool1,
    kernel_size=[5, 5],
    filters=64,
    strides=1,
    padding="same",
    activation=tf.nn.relu
)

pool2 = tf.layers.max_pooling2d(
    inputs=conv2,
    strides=2,
    pool_size=[2, 2]
)

flat = tf.reshape(
    pool2,
    [-1,7*7*64]
)

dense = tf.layers.dense(
    inputs=flat,
    units=1024,
    activation=tf.nn.relu
)

dropout = tf.layers.dropout(
    inputs=dense,
    rate=0.5
)

logits = tf.layers.dense(
    inputs=dropout,
    units=10
)

loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y,logits=logits)

train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

accuracy = tf.metrics.accuracy(
    labels=tf.argmax(output_y,axis=1),
    predictions= tf.argmax(logits,axis=1)
)[1]


sess = tf.Session()
# 初始化 全局和局部变量   局部变量在计算accuracy的时候用到
init = tf.group(
    tf.global_variables_initializer(),
    tf.local_variables_initializer()
)
sess.run(init)

for i in range(20000):
    #  每一批取50个样本
    batch = mnist.train.next_batch(50)
    train_loss, trian_op_ = sess.run(
        [loss, train_op],
        {input_x: batch[0], output_y: batch[1]}
    )
    if i % 100 ==0 :
        test_accuracy = sess.run(accuracy,{input_x:test_x,output_y:test_y})
        print("step:%d,loss:%f,accuracy:%f" % (i,train_loss,test_accuracy))
sess.close()
