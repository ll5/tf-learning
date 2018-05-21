# -*- coding: utf-8 -*-
import numpy as np

list1 = np.array([1,2,3,4])
list2 = np.array([5,6,7,8])
# print(list1.reshape(2,2))
# print(list2.reshape(2,2))

# print(list1.dot(list2))

print(np.hstack((list1,list2)))

print(np.split(list1,4))