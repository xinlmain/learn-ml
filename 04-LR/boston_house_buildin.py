import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 读取波士顿房价数据集
X, y = load_boston(return_X_y=True)

# 70% 用于训练，30% 用于测试
X0_train, X0_test, y_train, y_test = train_test_split(X, y, test_size =0.3, random_state = 0)

# 训练
# 通过最小二乘法的矩阵算法来训练
# 构造 X_train，即给 X0_train 增加一行 1
ones = np.ones(X0_train.shape[0]).reshape(-1, 1)
X_train = np.hstack((ones, X0_train))

pinvX_train = np.linalg.pinv(X_train) # 计算伪逆
w1 = np.matmul(pinvX_train, y_train) # 最小二乘法的矩阵算法
with np.printoptions(precision=3, suppress=True):	# 设置输出格式
	print('最小二乘法的矩阵算法找到的 w = {}'.format(w1))

# 通过 sklearn 的 LinearRegression 来训练
LR = LinearRegression().fit(X0_train, y_train)
w2 = np.insert(LR.coef_, 0, LR.intercept_, axis=0)
with np.printoptions(precision=3, suppress=True):	# 设置输出格式
	print('sklearn 找到的 w = {}'.format(w2))