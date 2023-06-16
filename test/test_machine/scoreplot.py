import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读入数据
data = pd.read_csv('./data/degree_tnm.csv', index_col=0)

# 计算皮尔逊相关系数
corr_matrix = data.corr(method='pearson')

# 筛选出前20个相关性最高的变量对
top_corr = corr_matrix.unstack().sort_values(ascending=False)[:20].reset_index()

# 提取相关性最高的变量名
cols = top_corr[['level_0', 'level_1']].values.flatten()

# 根据相关性最高的变量建立子集
subset = data.loc[:, cols]

# 计算距离矩阵
dist_matrix = 1 - subset.corr()

# 绘制层次聚类热力图
sns.set(font_scale=0.7)
g = sns.clustermap(subset, method='complete', metric='euclidean', figsize=(8, 8),
                   row_cluster=True, col_cluster=True, cmap='coolwarm', annot=True,
                   fmt='.2f', dendrogram_ratio=0.05, dendrogram_kwds={'color_threshold': 0})
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.show()


