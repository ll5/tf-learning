# -*- coding:utf-8 -*-
# 梯度下降解决线性回归问题

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


#build data

points_num = 100
vectors = []

for x in range(points_num):
    x1 = np.random.normal(0,1)
    y1 = 0.1 * x1 + 0.2 + np.random.normal(-0.02,0.02)
    vectors.append([x1,y1])
    pass

x_data = [v[0] for v in vectors]
y_data = [v[1] for v in vectors]

# show data
# plt.plot(x_data,y_data,'r*')
# plt.show()

w = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.zeros([1]))
y = w * x_data + b


loss = tf.reduce_mean(tf.square(y - y_data))

optimozer = tf.train.GradientDescentOptimizer(0.5)
train = optimozer.minimize(loss)

with tf.Session() as sess :
    init = tf.global_variables_initializer()
    sess.run(init)

    for step in range(20):
        sess.run(train)
        print("step:%d,loass:%f,bias:%f,weight:%f"%(step,sess.run(loss),sess.run(b),sess.run(w)))

    plt.plot(x_data,y_data,'r*')
    plt.plot(x_data,sess.run(w)*x_data+sess.run(b),'b-')
    plt.show()