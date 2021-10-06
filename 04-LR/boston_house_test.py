import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 读取波士顿房价数据集
X, y = load_boston(return_X_y=True)

# 70% 用于训练，30% 用于测试
X0_train, X0_test, y_train, y_test = train_test_split(X, y, test_size =0.3, random_state = 0)

# 训练
# 构造 X_train，即给 X0_train 增加一行 1
ones = np.ones(X0_train.shape[0]).reshape(-1, 1)
X_train = np.hstack((ones, X0_train))

pinvX_train = np.linalg.pinv(X_train) # 计算伪逆
w = np.matmul(pinvX_train, y_train) # 最小二乘法的矩阵算法

# 打印训练结果
with np.printoptions(precision=3, suppress=True):	# 设置输出格式
	print('训练结果：w = {} 。'.format(w))

# 测试泛化能力
# 构造 X_test，即给 X0_test 增加一行 1
ones = np.ones(X0_test.shape[0]).reshape(-1, 1)
X_test = np.hstack((ones, X0_test))

# 根据训练出来的 w ，给出预测房价 y_pred
y_pred = np.matmul(X_test, w) # 求出 y_pred

# 打印测试结果
print('测试集上的经验误差，即泛化误差为：{}'.format(mean_squared_error(y_pred, y_test)))