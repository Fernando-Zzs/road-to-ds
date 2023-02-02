# -*- coding = utf-8 -*-
# @Time : 2023/2/2 12:57
# @Author : Fernando
# @File : titanicCase.py
# @Software : PyCharm

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

from sklearn import tree
import graphviz

# 数据加载
train_data = pd.read_csv('./csv/train.csv')
test_data = pd.read_csv('./csv/test.csv')


def exploreData():
    # 数据探索
    print(train_data.info())
    print('-' * 30)
    print(train_data.describe())
    print('-' * 30)
    print(train_data.describe(include=['O']))
    print('-' * 30)
    print(train_data.head())
    print('-' * 30)
    print(train_data.tail())


"""
    Age、Fare 和 Cabin 这三个字段的数据有所缺失。
    其中 Age 为年龄字段，是数值型，我们可以通过平均值进行补齐；
    Fare 为船票价格，是数值型，我们也可以通过其他人购买船票的平均值进行补齐。
    Cabin 为船舱，有大量的缺失值。在训练集和测试集中的缺失率分别为 77% 和 78%，无法补齐；
    Embarked 为登陆港口，有少量的缺失值，我们可以使用登录最多的港口来把缺失值补齐。
"""


def washData():
    train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
    test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)
    train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
    test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)
    train_data['Embarked'].fillna('S', inplace=True)
    test_data['Embarked'].fillna('S', inplace=True)


def decisionTree():
    # 特征选择
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    train_features = train_data[features]
    train_labels = train_data['Survived']
    test_features = test_data[features]

    # 处理Embarked列的符号化对象S、C、Q
    dvec = DictVectorizer(sparse=False)
    # 特征向量转化为特征值矩阵
    train_features = dvec.fit_transform(train_features.to_dict(orient='record'))
    print(dvec.feature_names_)

    # 构造 ID3 决策树
    clf = DecisionTreeClassifier(criterion='entropy')
    # 决策树训练
    clf.fit(train_features, train_labels)

    test_features = dvec.transform(test_features.to_dict(orient='record'))
    # 决策树预测
    pred_labels = clf.predict(test_features)
    # 得到决策树准确率
    # acc_decision_tree = round(clf.score(train_features, train_labels), 6)
    # print(u'score 准确率为 %.4lf' % acc_decision_tree)
    print(u'cross_val_score 准确率为 %.4lf' % np.mean(
        cross_val_score(clf, train_features, train_labels, cv=10, scoring='accuracy')))
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render('output/decisionTree.gv', view=True)

if __name__ == '__main__':
    washData()
    decisionTree()