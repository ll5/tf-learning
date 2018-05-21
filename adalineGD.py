# -*- coding: utf-8 -*-
import  numpy as np
class AdalineGD(object):
    def __init__(self,eta=0.01,n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
        pass
    # 训练函数
    def fit(self,X,y):
        # 初始化权重向量为0
        self.w_ = np.zeros(1+X.shape[1])
        # 成本数值
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self
        pass
    def net_input(self,X):
        return np.dot(X,self.w_[1:]) + self.w_[0]
        pass
    def activation(self,X):
        return self.net_input(X)
        pass
    def predict(self,X):
        return np.where(self.activation(X)>=0,1,-1)
        pass
    pass
