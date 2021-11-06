import numpy as np
from sklearn import datasets
import datetime, math

# 需要设置的参数
wk = np.array([0, 0, 0])		# 初始点
eta = 0.1					    # 学习率
epochs = 156192 				# 迭代上限

# 鸢尾花数据集
sampleNumber = 100
iris = datasets.load_iris()
X = np.insert(iris.data[50:50+sampleNumber, [2,3]], 0, 1, axis=1)	# 取其中两种鸢尾花的最后两个特征，并给每行的第一列增加 1
y = np.where(iris.target[50:50+sampleNumber] == 1, -1, 1)			# 两种类别分别用标签 -1 和 1 来表示

# sigmoid 函数
def sigmoid(x):
    return 1/(1+np.exp(-x))

# 逻辑回归的假设函数
def h(x):
	return sigmoid(x@wk.T)

# 经验误差函数
def rhd(w):
	return np.mean(np.log(1+np.exp(-y*(X@w))))

# 经验误差函数的梯度
def drhd(w):
	ew0 = np.mean(-y*X[:, 0]*np.exp(-y*(X@w))/(1+np.exp(-y*(X@w))))
	ew1 = np.mean(-y*X[:, 1]*np.exp(-y*(X@w))/(1+np.exp(-y*(X@w))))
	ew2 = np.mean(-y*X[:, 2]*np.exp(-y*(X@w))/(1+np.exp(-y*(X@w))))
	return np.array([ew0, ew1, ew2])

# 经验误差函数的随机梯度
def srhd(w):
	# 从 X 和 y 中随机挑出一个点
	ran = np.random.randint(sampleNumber)
	xi, yi = X[ran], y[ran]
	# 对这个点求经验误差函数的梯度，没有直接用 drhd 函数是因为用矩阵速度太慢
	return -yi*xi*math.exp(-yi*np.dot(w, xi))/(1+math.exp(-yi*np.dot(w, xi)))

# 随机梯度下降法
starttime = datetime.datetime.now()
for _ in range(epochs):
	srhdwk = srhd(wk)
	wk = wk-eta*srhdwk

endtime = datetime.datetime.now()
print('迭代 {} 次后停止，||∇Rhd|| = {}，耗时约 {} 。'.format(epochs, np.linalg.norm(drhd(wk)), endtime-starttime))