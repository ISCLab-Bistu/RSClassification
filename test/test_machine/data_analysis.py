import pandas as pd
import numpy as np
from pymrmr import mRMR
from sklearn.decomposition import PCA

# from mrmr import mrmr_ensemble

# reading data
dataset = pd.read_csv('data/single_cell.csv', encoding='utf-8')
dataset = dataset.sample(frac=1.0)

# Initialize the PCA model and set the number of principal components to keep
pca_model = PCA(n_components=0.9)

X = dataset.iloc[:, 1:].values
# 对数据进行降维，得到新的特征矩阵
new_features = pca_model.fit_transform(X)

# 查看每个主成分的方差贡献比例
print(pca_model.explained_variance_ratio_)

# 保存数据
save_df = pd.DataFrame(new_features)
save_df.to_csv('./data/pca.csv', index=False)

# mr = mRMR(dataset, 'MID', 10)
# print(mr)

# 数据预处理
# X = dataset.iloc[:, 1:].values
# Y = dataset['labels'].values


# 绘图看下相关性
# y = list(range(X.shape[0]))
# plt.plot(y, X.iloc[:, 0:1])
# plt.plot(y, X.iloc[:, 1:2])
# # plt.plot(y, X.iloc[:, 4:5])
# plt.show()

# 使用皮尔逊相关系数
# corr_data = dataset.corr()
# corr_data = corr_data.values
# print(type(corr_data))
# # 提取出相关性大于0.9的数值
# for i in range(X.shape[1]):
#     for j in range(X.shape[1]):
#         if i == 0:
#             print(j)
#             print(corr_data[i][j])
#
#         # if corr_data[i][j] >= 0.7:
#         #     if i != j:
#         #         print(i)
#         #         print(j)
#         #         print("################")
#
# sns.heatmap(corr_data)
# plt.show()

# 对相关性高的数据进行降维(0, 1)
# X = dataset.values
# pca_column = [8, 9]  # 这里需要从1开始
# data1 = X[:, pca_column]
# pca = PCA(n_components=1)
# new_data1 = pca.fit_transform(data1)
# print(new_data1.shape)
#
# pca_column = [21, 24]  # 这里需要从1开始
# data2 = X[:, pca_column]
# pca = PCA(n_components=1)
# new_data2 = pca.fit_transform(data2)
#
# pca_column = [25, 27, 28]  # 这里需要从1开始
# data3 = X[:, pca_column]
# pca = PCA(n_components=1)
# new_data3 = pca.fit_transform(data3)
#
# # pca_column = [35, 37]  # 这里需要从1开始
# # data4 = X[:, pca_column]
# # pca = PCA(n_components=1)
# # new_data4 = pca.fit_transform(data4)
#
# # pca_column = [17, 23]  # 这里需要从1开始
# # data5 = X[:, pca_column]
# # pca = PCA(n_components=1)
# # new_data5 = pca.fit_transform(data5)
#
# pca_column = [8, 9, 21, 24, 25, 27, 28]
#
# X = np.delete(X, pca_column, axis=1)
# X = np.append(X, new_data1, axis=1)
# X = np.append(X, new_data2, axis=1)
# X = np.append(X, new_data3, axis=1)
# # X = np.append(X, new_data4, axis=1)
# # X = np.append(X, new_data5, axis=1)
# print(X.shape)
#
# data_df = pd.DataFrame(X)
# print(data_df)
# data_df.to_csv("./data/test_three.csv", index=False)
