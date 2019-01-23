from KNN.py import KNNClassfilter
import numpy as np

'''
    测试脚本
'''

data_x = [[3.393,2.331],
       [3.110,1.781],
       [1.343,3.368],
       [3.582,4.679],
       [2.280,2.866],
       [7.423,4.696],
       [5.745,3.533],
       [9.172,2.511],
       [7.792,3.424],
       [7.939,0.791]]
       
data_y = [0,0,0,0,0,1,1,1,1,1]
x_train = np.array(data_x)
y_train = np.array(data_y)
x = np.array([9.888, 3.555])
# 要将x这个矩阵转换成二维的矩阵，一行两列的矩阵
x_predict = x.reshape(1, -1)

# 创建一个对象，k值为6
knn_clf = KNNClassfilter(6)
# 将训练数据和训练标签拟合
knn_clf.fit(x_train, y_train)
# 输入待预测数据
y_predict = knn_clf.predict(x_predict)
print(y_predict)
