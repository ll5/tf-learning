# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# var1 = tf.constant([[1,2],[2,1]])
# var2 = tf.constant([[3,4],[4,3]])
#
# # sess = tf.Session()
# # sess.run(var)
# # sess.close()
# with tf.Session() as sess:
#     mtp = tf.matmul(var1,var2)
#     print(mtp)
#     res = sess.run(mtp)
#     if sess.graph is tf.get_default_graph():
#         print('是默认图')
#     print(res)
