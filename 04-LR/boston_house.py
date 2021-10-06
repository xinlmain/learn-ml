import numpy as np
from sklearn.datasets import load_boston

# 读取波士顿房价数据集
X0, y = load_boston(return_X_y=True)

# 构造 X，即给 X0 增加一列 1
ones = np.ones(X0.shape[0]).reshape(-1, 1)
X = np.hstack((ones, X0))

pinvX = np.linalg.pinv(X)   # 计算伪逆
w = pinvX @ y               # 最小二乘法的矩阵算法

# 打印结果
with np.printoptions(precision=3, suppress=True):	# 设置输出格式
	print('结果：w = {} 。'.format(w))