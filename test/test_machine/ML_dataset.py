import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
dataset = pd.read_csv('./data/ovarian_cancer.csv', encoding='utf-8')
dataset = dataset.sample(frac=1.0)
# print(dataset)

X = dataset.iloc[:, 1:].values
Y = dataset['labels'].values

# Calculating the correlation matrix
# corr_matrix = dataset.corr()
# print(corr_matrix)

# Visualizing the correlation matrix
# sns.heatmap(corr_matrix, annot=False)
# plt.show()

# Select the features with strong correlation
# selected_features = corr_matrix[abs(corr_matrix['labels']) > 0.09].index.tolist()
# print(selected_features)

# Extract the selected features as well as the target variable
#
# X = dataset[selected_features].iloc[:, 1:].values
# print(X.shape)
# Y = dataset[selected_features].iloc[:, 0:1].values

# Initialize the PCA model and set the number of principal components to keep
pca_model = PCA(n_components=0.9)
X = pca_model.fit_transform(X)

accuracy_top = 0
precision = 0
recall = 0
f1_score = 0
# K-fold partition subset
split = 6
KF = KFold(n_splits=split)
for train_index, test_index in KF.split(X):
    # print("TRAIN", train_index, "TEST", test_index)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # Data normalization feature scaling
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from xgboost.sklearn import XGBClassifier
    from sklearn.linear_model import Perceptron
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression

    # classifier = RandomForestClassifier(n_estimators=10, class_weight=None)
    # classifier = SVC(C=20, kernel='linear', class_weight=None, probability=True)
    classifier = XGBClassifier(class_weight=None)
    # classifier = Perceptron(class_weight=None, shuffle=False, fit_intercept=False)
    # classifier = KNeighborsClassifier(n_neighbors=3, class_weight=class_weight)
    # classifier = DecisionTreeClassifier(max_depth=5, class_weight=class_weight)
    # classifier = GaussianNB()
    # classifier = LogisticRegression()

    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)

    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

    accuracy = accuracy_score(Y_test, y_pred)
    accuracy_top += accuracy
    result1 = precision_score(Y_test, y_pred, average='macro')
    result2 = recall_score(Y_test, y_pred, average='macro')
    precision += result1
    recall += result2

print("Accuracy:{}".format(accuracy_top / split))
print("Precision:{}".format(precision / split))
print("Recall:{}".format(recall / split))
