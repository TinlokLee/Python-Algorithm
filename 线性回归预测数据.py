'''
    线性回归预测数据
    
    1. 预测房价
        必须在数据中找出一种线性关系
        数据存储成一个.csv文件
        把X_parameter和Y_parameter拟合为线性回归模型
        验证

    2. 预测观影数
        预测
        写入文件

    3. 替换数据集中缺失值
        首先我们找到我们要替换那一列里的缺失值，并找出缺失值依赖于其他列的
        哪些数据。把缺失值那一列作为Y_parameters，把缺失值更依赖的那些列
        作为X_parameters，并把这些数据拟合为线性回归模型。现在就可以用
        缺失值更依赖的那些列预测缺失的那一列

    开发环境：
    安装包：
        Numpy: pip install python-numpy
               sudo apt-get update 
               sudo apt-get install python-numpy
               from numpy import *
               a = arange(12)
               a = a.reshape(3,2,2)
               print a

        Pandas:pip install python-pandas
            import pandas as pd
            values = np.array([2.0, 1.0, 3.0, 10.0, 0.0599, 8.0])
            ser = pd.Series(values)
            print ser
        
        SciPy: pip install python-scipy
            from scipy import special, optimize
            f = lambda x: -special.jv(3, x)
            sol = optimize.minimize(f, 1.0)
            x = linspace(0, 10, 5000)
            plot(x, special.jv(3, x), '-', sol.x, -sol.fun, 'o')
            savefig('plot.png', dpi=96)

        Matplotlib: pip install install python-matplotlib
            import numpy as np
            import matplotlib.mlab as mlab
            import matplotlib.pyplot as plt
 
            mu = 100 # mean of distribution
            sigma = 15 # standard deviation of distribution
            x = mu + sigma * np.random.randn(10000)
 
            num_bins = 50
            n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)
            y = mlab.normpdf(bins, mu, sigma)
            plt.plot(bins, y, 'r--')
            plt.xlabel('Smarts')
            plt.ylabel('Probability')
            plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')
 
            plt.subplots_adjust(left=0.15)
            plt.show()

        IPython: pip install ipython
            # 这段代码用于绘制积分作为曲线下面积的曲线
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.patches import Polygon
            def func(x):
                return (x - 3) * (x - 5) * (x - 7) + 85
 
            a, b = 2, 9 # integral limits
            x = np.linspace(0, 10)
            y = func(x)
 
            fig, ax = plt.subplots()
            plt.plot(x, y, 'r', linewidth=2)
            plt.ylim(ymin=0)
 
            ix = np.linspace(a, b)
            iy = func(ix)
            verts = [(a, 0)] + list(zip(ix, iy)) + [(b, 0)]
            poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
            ax.add_patch(poly)
 
            plt.text(0.5 * (a + b), 30, r"$\int_a^b f(x)\mathrm{d}x$",
            horizontalalignment='center', fontsize=20)
 
            plt.figtext(0.9, 0.05, '$x$')
            plt.figtext(0.1, 0.9, '$y$')
 
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
 
            ax.set_xticks((a, b))
            ax.set_xticklabels(('$a$', '$b$'))
            ax.set_yticks([])
 
            plt.show()
    
    scikit:  pip install python-sklearn
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn import datasets, linear_model
 
        diabetes = datasets.load_diabetes()
 
        diabetes_X = diabetes.data[:, np.newaxis]
        diabetes_X_temp = diabetes_X[:, :, 2]
 
        diabetes_X_train = diabetes_X_temp[:-20]
        diabetes_X_test = diabetes_X_temp[-20:]
 
        # Split the targets into training/testing sets
        diabetes_y_train = diabetes.target[:-20]
        diabetes_y_test = diabetes.target[-20:]
 
        # Create linear regression object
        regr = linear_model.LinearRegression()
 
        # Train the model using the training sets
        regr.fit(diabetes_X_train, diabetes_y_train)
 
        # The coefficients
        print('Coefficients: \n', regr.coef_)
        # The mean square error
        print("Residual sum of squares: %.2f"
        % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

        # Plot outputs
        plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
        plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
        linewidth=3)
 
        plt.xticks(())
        plt.yticks(())
 
        plt.show()

        系数：
        [938.23786125] 
        剩余平方和：2548.07 
        差异分数：0.47

'''
# Required Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model


def get_data(file_name):
    # 将.csv数据读入Pandas数据帧
    data = pd.read_csv(file_name)
    X_parameter = []
    Y_parameter = []
    for single_square_feet ,single_price_value in zip(data['square_feet'],data['price']):
        # 把Pandas数据帧转换为X_parameter和Y_parameter数据
        X_parameter.append([float(single_square_feet)])
        Y_parameter.append(float(single_price_value))
    return X_parameter,Y_parameter

def linear_model_main(X_parameters,Y_parameters,predict_value):
    regr = linear_model.LinearRegression()
    # 创建一个线性模型，用我们的X_parameters和Y_parameter训练
    regr.fit(X_parameters, Y_parameters)
    predict_outcome = regr.predict(predict_value)

    # 创建predictions字典，存着θ0、θ1和预测值，并返回predictions字典为输出
    predictions = {}
    predictions['intercept'] = regr.intercept_
    predictions['coefficient'] = regr.coef_
    predictions['predicted_value'] = predict_outcome
    return predictions


if __name__ == "__main__":
    X,Y = get_data('input_data.csv')
    predictvalue = 700
    result = linear_model_main(X,Y,predictvalue)
    print("Intercept value " , result['intercept'])
    print("coefficient" , result['coefficient'])
    print("Predicted value: ",result['predicted_value'])

    # Intercept value（截距值）就是θ0的值，
    # coefficient value（系数）就是θ1的值。 
    # 我们得到预测的价格值为21915.4255



# 验证数据怎么拟合线性回归,定义函数，输入为X_parameters和Y_parameters，
# 显示出数据拟合的直线
def show_linear_line(X_parameters,Y_parameters):
    regr = linear_model.LinearRegression()
    regr.fit(X_parameters, Y_parameters)
    plt.scatter(X_parameters,Y_parameters,color='blue')
    plt.plot(X_parameters,regr.predict(X_parameters),color='red',linewidth=4)
    plt.xticks(())
    plt.yticks(())
    plt.show()

    # 调用函数
    show_linear_line(X,Y)



# 观影数预测
import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model


def get_data(file_name):
    """
    写一个函数，把我们的数据集作为输入，
    返回flash_x_parameter、flash_y_parameter、
    arrow_x_parameter、arrow_y_parameter values
    """
    data = pd.read_csv(file_name)
    flash_x_parameter = []
    flash_y_parameter = []
    arrow_x_parameter = []
    arrow_y_parameter = []
    for x1,y1,x2,y2 in zip(data['flash_episode_number'],\
                data['flash_us_viewers'],data['arrow_episode_number'],\
                data['arrow_us_viewers']):
        flash_x_parameter.append([float(x1)])
        flash_y_parameter.append(float(y1))
        arrow_x_parameter.append([float(x2)])
        arrow_y_parameter.append(float(y2))
    return flash_x_parameter,flash_y_parameter,arrow_x_parameter,arrow_y_parameter

def more_viewers(x1,y1,x2,y2):
    """
    预测
    """
    regr1 = linear_model.LinearRegression()
    regr1.fit(x1, y1)
    predicted_value1 = regr1.predict(9)
    print(predicted_value1)
    regr2 = linear_model.LinearRegression()
    regr2.fit(x2, y2)
    predicted_value2 = regr2.predict(9)
    #print predicted_value1
    #print predicted_value2
    if predicted_value1 > predicted_value2:
        print("The Flash Tv Show will have more viewers for next week")
    else:
        print("Arrow Tv Show will have more viewers for next week")



# 写入文件
import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
  

def get_data(file_name):
    data = pd.read_csv(file_name)
    flash_x_parameter = []
    flash_y_parameter = []
    arrow_x_parameter = []
    arrow_y_parameter = []
    for x1,y1,x2,y2 in zip(data['flash_episode_number'],\
                data['flash_us_viewers'],data['arrow_episode_number'],\
                data['arrow_us_viewers']):
        flash_x_parameter.append([float(x1)])
        flash_y_parameter.append(float(y1))
        arrow_x_parameter.append([float(x2)])
        arrow_y_parameter.append(float(y2))
    return flash_x_parameter,flash_y_parameter,arrow_x_parameter,arrow_y_parameter
  

def more_viewers(x1,y1,x2,y2):
    regr1 = linear_model.LinearRegression()
    regr1.fit(x1, y1)
    predicted_value1 = regr1.predict(9)
    print(predicted_value1)
    regr2 = linear_model.LinearRegression()
    regr2.fit(x2, y2)
    predicted_value2 = regr2.predict(9)

    if predicted_value1 > predicted_value2:
        print("The Flash Tv Show will have more viewers for next week")
    else:
        print("Arrow Tv Show will have more viewers for next week")
  
x1,y1,x2,y2 = get_data('input_data.csv')
#print x1,y1,x2,y2
more_viewers(x1,y1,x2,y2)