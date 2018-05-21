# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


file = './data.csv'
df = pd.read_csv(file,header=None)

y = df.loc[0:100,4].values
y = np.where(y == 'Iris-setosa',-1 ,1)
x = df.loc[0:100,[0,2]].values

plt.scatter(x[:50,0],x[:50,1],color='green',marker='o',label='setosa')
plt.scatter(x[50:100,0],x[50:100,1],color='red',marker='x',label='versicolor')
plt.xlabel('flower length')
plt.ylabel('base length')
plt.legend(loc='upper left')
plt.show()

