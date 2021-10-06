# # 从sklearn库中引入美国威斯康辛州的癌症数据集
# from sklearn.datasets import load_breast_cancer
# # 引入numpy，方便我们进行数组操作
# import numpy as np
#
# # 获取特征向量集合，以及标签向量集合
# X, y = load_breast_cancer(return_X_y=True)
#
# print('该数据库中总共有 {} 位病人（特征向量）。'.format(X.shape[0]))
# print('每位病人（特征向量）包含 {} 个特征。'.format(X.shape[1]))
# print('所有病人有 {} 种标签，分别是“良性肿瘤”、“恶性肿瘤”。'.format(np.size(np.unique(y))))

import json

import pandas as pd

#js = json.load("~/top1k.json")

top = pd.read_json("~/top1k.json", orient="records")
top["cate"] = top["_source"].apply(lambda l: l["cat_name"])

sorted = pd.DataFrame({'total': top.groupby("cate").size()})
sorted = sorted.sort_values(['total'], ascending=False)

print(sorted)
# print(top)