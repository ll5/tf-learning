# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 导入数据
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist_data', one_hot=True)

# 声明 3 个 placeholder
input_x = tf.placeholder(tf.float32, [None, 28 * 28]) / 255.0
output_y = tf.placeholder(tf.int32, [None, 10])
input_x_images = tf.reshape(input_x, [-1, 28, 28, 1])

# 获取3000个测试数据
test_x = mnist.test.images[:3000]
test_y = mnist.test.labels[:3000]

# 第一层卷积
conv1 = tf.layers.conv2d(
    inputs=input_x_images,
    filters=32,
    kernel_size=[5, 5],
    strides=1,
    padding="same",
    activation=tf.nn.relu,
)  # 输出形状 [28 * 28 * 32]

# 第一层池化
pool1 = tf.layers.max_pooling2d(
    inputs=conv1,
    strides=2,
    pool_size=[2, 2],
)  # 输出形状 [14 * 14 * 32]

# 第二层卷积
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    strides=1,
    padding="same",
    activation=tf.nn.relu,
)  # 输出形状 [14 * 14 * 64]

# 第二层池化
pool2 = tf.layers.max_pooling2d(
    inputs=conv2,
    strides=2,
    pool_size=[2, 2],
)  # 输出形状 [7 * 7 * 64]

# 扁平化
flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

# 1024 个神经元的全连接层
dense = tf.layers.dense(
    inputs=flat,
    units=1024,
    activation=tf.nn.relu
)
# 丢弃掉 50% (丢弃率0.5)
dropout = tf.layers.dropout(
    inputs=dense,
    rate=0.5,
)

# 10 个神经元的全连接层
logits = tf.layers.dense(
    inputs=dropout,
    units=10,

)  # 输出形状 [1 * 1 * 10]

# 计算误差 (交叉熵)
loss = tf.losses.softmax_cross_entropy(
    onehot_labels=output_y,
    logits=logits,
)

# 用Adam优化器,最小化误差
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# 计算预测值 和实际标签 的匹配度
accuracy = tf.metrics.accuracy(
    labels=tf.argmax(output_y, axis=1),
    predictions=tf.argmax(logits, axis=1)
)[1]

# 训练神经网络
with tf.Session() as sess:
    init = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    sess.run(init)

    for step in range(200):
        # 每一批取 50 个训练数据
        batch = mnist.train.next_batch(50)
        train_loss,train_op_ = sess.run(
            [loss, train_op],
            {input_x: batch[0], output_y: batch[1]}
        )

        if step % 2 == 0:
            test_accuracy = sess.run(
                accuracy,
                {input_x:test_x,output_y:test_y}
            )

            print("第%d次训练,loss=%f,accuracy=%f" % (step,train_loss,test_accuracy))

        test_output = sess.run(logits, {input_x:test_x[:20]})
        inferenced_y = np.argmax(test_output,1)

    print("推测数据" % inferenced_y)
    print("真实数据" % np.argmax(test_y[:20],1))