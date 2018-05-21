# -*- coding:utf-8 -*-

# 激活函数

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#build data

x = np.linspace(-7,7,180)

def sigmoid(inputs):
    y = [1 / float(1+np.exp(-x)) for x in inputs]
    return y

def relu (inputs) :
    y = [x * (x>0) for x in inputs]
    return y

def tanh(inputs):
    y = [(np.exp(x) - np.exp(-x)) / float(np.exp(x) + np.exp(-x)) for x in inputs]
    return y

def softplus(inputs):
    y = [np.log(1+ np.exp(x)) for x in inputs]
    return y
plt.subplot(221)
plt.plot(x,sigmoid(x),label="sigmoid")
plt.legend(loc="best")
plt.subplot(222)
plt.plot(x,relu(x),label="relu")
plt.legend(loc="best")
plt.subplot(223)
plt.plot(x,tanh(x),label="tanh")
plt.legend(loc="best")
plt.subplot(224)
plt.plot(x,softplus(x),label="softplus")
plt.legend(loc="best")

plt.show()