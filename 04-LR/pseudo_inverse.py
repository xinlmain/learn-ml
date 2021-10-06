import numpy as np

# 父子身高数据集
X0 = np.array([1.51, 1.64, 1.6, 1.73, 1.82, 1.87]).reshape(-1, 1)
y = np.array([1.63, 1.7, 1.71, 1.72, 1.76, 1.86])

# 构造 X，即给 X0 增加一列 1
ones = np.ones(X0.shape[0]).reshape(-1, 1)
X = np.hstack((ones, X0))

pinvX = np.linalg.pinv(X)   # 计算伪逆
w = pinvX @ y               # 最小二乘法的矩阵算法

# 打印结果
print('最小二乘法的矩阵算法的结果为：w0 = {}, w1 = {}。'.format(w[0], w[1]))