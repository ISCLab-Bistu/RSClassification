# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
import pandas as pd
from xgboost import XGBClassifier

# 读取数据
dataset = pd.read_csv('data/augument_ovarian.csv', encoding='utf-8')
# dataset = dataset.sample(frac=1.0)
train_df = dataset.iloc[387:].sample(frac=1.0)
test_df = dataset.iloc[0:386].sample(frac=1.0)

# print(train_df)

# 数据预处理
# X = dataset.iloc[:, 1:].values
# Y = dataset['labels'].values

# X_train, X_test, y_train, y_test = train_test_split(
#     X, Y, test_size=0.3)  # 按照比例划分数据集为训练集与测试集

X_train = train_df.iloc[:, 1:].values
y_train = train_df['labels']
X_test = test_df.iloc[:, 1:].values
y_test = test_df['labels']

# print(y_test)

# 数据标准化 特征标度
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建一个SVM分类器并进行预测````````````````````````
# classifier = RandomForestClassifier(n_estimators=20, criterion='entropy')
classifier = SVC(C=1, kernel='linear', gamma='auto')
# classifier = XGBClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score

result = accuracy_score(y_test, y_pred)
print(result)

# 创建一个感知机分类器并进行预测
# clf1 = Perceptron()  # 创建感知机训练模型
# clf1.fit(X_train, y_train)  # 队训练集数据进行训练
# clf1_y_predict = clf1.predict(X_test)  # 通过测试集数据，得到测试标签
# scores1 = clf1.score(X_test, y_test)  # 测试结果打分
#
# # 打印
# print('SVM准确率：', scores)
# print('感知机准确率：', scores1)

# K折交叉验证模块
# from sklearn.model_selection import cross_val_score
#
# # 使用K折交叉验证模块
# scores = cross_val_score(classifier, X, Y, cv=10, scoring='accuracy')
# # 将10次的预测准确率打印出
# print(scores)
#
# # 将10次的预测准确平均率打印出
# print(scores.mean())
