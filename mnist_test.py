# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist-data", one_hot=True)

input_x = tf.placeholder(tf.float32, [None, 28 * 28]) / 255.
output_y = tf.placeholder(tf.int32, [None, 10])
input_images = tf.reshape(input_x, [-1, 28, 28, 1])

# 构建

conv1 = tf.layers.conv2d(
    inputs=input_images,
    strides=1,
    kernel_size=[5, 5],
    padding='same',
    activation=tf.nn.relu,
    filters=32
)

pool1 = tf.layers.max_pooling2d(
    inputs=conv1,
    strides=2,
    pool_size=[2, 2]
)

conv2 = tf.layers.conv2d(
    inputs=pool1,
    strides=1,
    kernel_size=[5, 5],
    padding='same',
    activation=tf.nn.relu,
    filters=64
)

pool2 = tf.layers.max_pooling2d(
    inputs=conv2,
    strides=2,
    pool_size=[2, 2]
)

flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
dense = tf.layers.dense(
    inputs=flat,
    units=1024,
    activation=tf.nn.relu
)
dropout = tf.layers.dropout(inputs=dense,rate=0.5)

logits = tf.layers.dense(
    inputs=dropout,
    units=10
)

loss = tf.losses.softmax_cross_entropy(logits=logits,onehot_labels=output_y)

train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)


# 计算精度
accuracy =tf.metrics.accuracy(
    labels=tf.argmax(output_y,axis=1),
    predictions=tf.argmax(logits,axis=1)
)[1]

sess = tf.Session()
init = tf.group(
    tf.global_variables_initializer(),
    tf.local_variables_initializer()
)
sess.run(init)

test_x = mnist.test.images[:3000]
test_y = mnist.test.labels[:3000]

for i in range(500):
    batch = mnist.train.next_batch(50)
    train_loss,train_op_ =sess.run([
        loss,train_op],
        {input_x:batch[0],output_y:batch[1]}
    )

    if i % 100 == 0:
        test_accuracy = sess.run(accuracy,{input_x:test_x,output_y:test_y})
        print("step:%d,loss:%4f,accuracy:%4f," % (i,train_loss,test_accuracy))

# 打印20个比对结果

test_output = sess.run(logits,{input_x:test_x[:20]})

print("实际结果", np.argmax(test_y[:20],1))
print("推测结果", np.argmax(test_output,1))
