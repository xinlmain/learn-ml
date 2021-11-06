import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 需要设置的参数
wk = np.array([0, 0, 0])		# 初始点
eta = 0.1					    # 学习率
epochs = 5000 				    # 迭代上限
epsilon = 0.01					# 梯度模长的上限

# 信用卡发放数据集
X = np.insert(np.array([[5,2], [3,2], [2,7], [1,4], [6,1], [4,5], [2,4.5]]), 0, 1, axis=1)	# 给每个xi第一列增加1
y = np.array([-1, -1, 1, 1, -1, 1, -1])

# sigmoid 函数
def sigmoid(x):
    return 1/(1+np.exp(-x))

# 逻辑回归的假设函数
def h(x):
	return sigmoid(x@wk.T)

# 经验误差函数的梯度
def drhd(w):
	ew0 = np.mean(-y*X[:, 0]*np.exp(-y*(X@w))/(1+np.exp(-y*(X@w))))
	ew1 = np.mean(-y*X[:, 1]*np.exp(-y*(X@w))/(1+np.exp(-y*(X@w))))
	ew2 = np.mean(-y*X[:, 2]*np.exp(-y*(X@w))/(1+np.exp(-y*(X@w))))
	return np.array([ew0, ew1, ew2])

# 梯度下降法
for i in range(epochs):
	drhdwk = drhd(wk)
	if np.linalg.norm(drhdwk) < epsilon:
		break
	wk = wk-eta*drhdwk

# 下面是绘制代码
# 设置字体大小
plt.rcParams.update({'font.size': 14})
# 设置subfigure
fig, ax = plt.subplots(figsize = (6, 3))
fig.subplots_adjust(left=0.25, right=0.75, top=0.999, bottom=0.001)
ax.set(xticks=[], yticks=[])

x1_min, x1_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
x2_min, x2_max = X[:, 2].min() - 0.2, X[:, 2].max() + 0.2
ax.set_xlim(x1_min, x1_max),ax.set_ylim(x2_min, x2_max)

# 绘制决策区域
resolution = 0.03
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
xx = np.insert(np.c_[xx1.ravel(), xx2.ravel()], 0, 1, axis=1)
Z = h(xx).reshape(xx1.shape)
ax.contourf(xx1, xx2, Z, 50, cmap="coolwarm", vmin=0, vmax=1, alpha=0.9)

# 绘制数据集
markers = ('x', 'o')
vmin, vmax = min(y), max(y)
cm = ListedColormap(('blue', 'red'))
for cl, m in zip(np.unique(y), markers):
	ax.scatter(x=X[y==cl, 1], y=X[y==cl, 2], c=y[y==cl], alpha=1, vmin = vmin, vmax = vmax, cmap=cm, edgecolors='k', marker = m)

# 绘制直线
def lh(x):
	return -wk[0]/wk[2]-wk[1]/wk[2]*x
xx1 = np.arange(x1_min, x1_max, 0.1)
ax.plot(xx1, lh(xx1), 'k--', lw=1)

plt.show()