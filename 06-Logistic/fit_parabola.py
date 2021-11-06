import numpy as np

# 需要设置的参数
xk = np.array([10])			    # 初始点
eta = 0.1 					    # 学习率
epochs = 5000 				    # 迭代上限
epsilon = 0.0000001				# 梯度模长的上限

# 抛物线函数
def f(x):
	return x[0]*x[0]

# 抛物线函数的梯度
def df(x):
	return 2*x

# 梯度下降法
for i in range(epochs):
	dfxk = df(xk)
	if np.linalg.norm(dfxk) < epsilon:
		print('经过 {} 次迭代，梯度下降法运行完毕'.format(i+1))
		print('结果为 xk = {} ，f(xk) = {}'.format(xk, f(xk)))
		break
	xk = xk-eta*dfxk