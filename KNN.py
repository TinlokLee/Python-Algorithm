'''
    scikit-learn库中 KNN 算法的封装与使用方法

    机器学习中，KNN 是不需要训练过程的算法，输入样例直接调用predict 预测结果
    模型： 训练数据集（训练数据和训练标签进行拟合形成模型）


'''
import numpy as np 
from math import sqrt
from collections import Counter 

class KNNClassfilter:
    def __init__(self, k):
        # 初始化KNN 分类器，并断言判断是否合法
        assert k >= 1
        self.k = k
        self._x_train = None
        self._y_train = None

    def fit(self, x_train, y_train):
        # 根据训练集x_train,y_train训练KNN分类器，形成模型
        # 数据和标签大小必须一样
        assert x_train.shape[0] == y_train.shape[0]
        # k 值不能超过数据的大小
        assert self.k <= x_train.shape[0]
        self._x_train = x_train
        self._y_train = y_train
        return self

    def predict(self, x_predict):
        # 必须将训练集和标签拟合为模型才能进行预测
        # 训练数据和标签不能为空
        assert self._x_train is not None and self._y_train is not None
        
        # 特征个数必须相等（待测数据和训练数据的列）
        assert x_predict.shape[1] == self._x_train.shape[1]
        y_predict = [self._predict(x) for x in x_predict]
        return np.array(y_predict)

    def _predict(self, x):
        # 给定单个待测数据x，返回x 的预测数据结果
        # x表示一行数据，即数组,它的特征数据个数，必须和训练数据相同
        assert x.shape[0] == self._x_train.shape[1]
        distances = [sqrt(np.sum((x_train - x)**2)) for x_train in self._x_train]
        nearest = np.argsort(distances)
        topk_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topk_y)
        return votes.most_common(1)[0][0]

# <== 新建测试脚本test.py, 引入KNNClassfilter对象