import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

rnd = np.random.RandomState(3)  # 为了演示，采用固定的随机
x_min, x_max = -1, 1


# 上帝函数 y=f(x)
def f(x):
    return x ** 2


# 上帝分布 P(Y|X)
def P(X):
    return f(X) + rnd.normal(scale=0.1, size=X.shape)


# 通过 P(X, Y) 生成数据集 D
X = rnd.uniform(x_min, x_max, 10)  # 通过均匀分布产生 X
y = P(X)  # 通过 P(Y|X) 产生 y

# 设置字体大小
plt.rcParams.update({'font.size': 14})
# 设置subfigure
fig, ax = plt.subplots(figsize=(6, 3))
plt.subplots_adjust(left=0.25, right=0.75, top=0.999, bottom=0.08)
ax.set(xticks=[], yticks=[])
ax.set_xlabel('$x$'), ax.set_ylabel('$y$')
ax.set_xlim(x_min, x_max)

# 绘制数据集
ax.scatter(x=X, y=y)

# 进行二次多项式回归
poly = PolynomialFeatures(degree=7)  # 建立二次多项式的坐标转换
Z = poly.fit_transform(X.reshape(-1, 1))  # 将 X 转化为 Z
LR = LinearRegression().fit(Z, y)  # 在 zy 坐标系中进行线性回归

# 在 xy 坐标中绘制二次曲线
xx = np.linspace(x_min, x_max)
zz = poly.fit_transform(xx.reshape(-1, 1))
ax.plot(xx, LR.predict(zz), 'k-')

plt.show()
