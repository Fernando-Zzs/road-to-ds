# -*- coding = utf-8 -*-
# @Time : 2023/1/25 19:56
# @Author : Fernando
# @File : learnPandas.py
# @Software : PyCharm

import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from pandasql import sqldf


def testCreateSeries():
    x1 = Series([1, 2, 3, 4])  # 以python列表形式创建
    x2 = Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])  # 指定索引，默认索引为0123
    print(x1)
    print(x2)
    d = {'a': 1, 'b': 2, 'c': 3}
    x3 = Series(d)  # 以字典形式创建
    print(x3)


def testCreateDataframe():
    data = {'Chinese': [66, 95, 93, 90, 80], 'English': [65, 85, 92, 88, 90], 'Math': [100, 99, 98, 97, 96]}
    df1 = DataFrame(data)
    df2 = DataFrame(data, index=['ZhangFei', 'GuanYu', 'ZhaoYun', 'HuangZhong', 'DianWei'])
    print(df1)
    print(df2)


def testImportAndExportData():
    score = DataFrame(pd.read_excel('./xlsx/data.xlsx'))
    score.to_excel('./xlsx/data1.xlsx')
    print(score)


def testDeletePortion():
    data = {'Chinese': [66, 95, 93, 90, 80], 'English': [65, 85, 92, 88, 90], 'Math': [100, 99, 98, 97, 96]}
    df = DataFrame(data, index=['ZhangFei', 'GuanYu', 'ZhaoYun', 'HuangZhong', 'DianWei'])
    df = df.drop(columns=['Chinese'])  # 删掉指定列
    df = df.drop(index=['ZhangFei'])  # 删掉指定行
    print(df)


def testRenameColumn():
    data = {'Chinese': [66, 95, 93, 90, 80], 'English': [65, 85, 92, 88, 90], 'Math': [100, 99, 98, 97, 96]}
    df = DataFrame(data, index=['ZhangFei', 'GuanYu', 'ZhaoYun', 'HuangZhong', 'DianWei'])
    df.rename(columns={'Chinese': 'YuWen', 'English': 'YingYu'}, inplace=True)
    print(df)


def testRemoveDuplicateRow():
    data = {'Chinese': [66, 66, 93, 90, 80], 'English': [65, 65, 92, 88, 90], 'Math': [100, 100, 98, 97, 96]}
    df = DataFrame(data, index=['ZhangFei', 'ZhangFei', 'ZhaoYun', 'HuangZhong', 'DianWei'])
    df = df.drop_duplicates()
    print(df)


def testChangeFormat():
    data = {'Chinese': [66, 95, 93, 90, 80], 'English': [65, 85, 92, 88, 90], 'Math': [100, 99, 98, 97, 96]}
    df = DataFrame(data, index=['ZhangFei', 'GuanYu', 'ZhaoYun', 'HuangZhong', 'DianWei'])
    df['Chinese'].astype('str')
    df['Chinese'].astype(np.int64)


def testRemoveSpaces():
    data = {'Chinese': ['$66 ', '$95 ', '$93 ', '$90 ', '$80 '], 'English': [' 65 ', ' 85 ', ' 92 ', ' 88 ', ' 90 '],
            'Math': [' 100 ', ' 99 ', ' 98 ', ' 97 ', ' 96 ']}
    df = DataFrame(data, index=['ZhangFei', 'GuanYu', 'ZhaoYun', 'HuangZhong', 'DianWei'])
    # 删除左右两边指定的字符
    df['Chinese'] = df['Chinese'].str.strip('$')
    # 删除左边空格
    df['English'] = df['English'].map(str.lstrip)
    # 删除右边空格
    df['Math'] = df['Math'].map(str.rstrip)
    print(df)


def testChangeCase():
    data = {'Chinese': [66, 95, 93, 90, 80], 'English': [65, 85, 92, 88, 90], 'Math': [100, 99, 98, 97, 96]}
    df = DataFrame(data, index=['ZhangFei', 'GuanYu', 'ZhaoYun', 'HuangZhong', 'DianWei'])
    # 全部大写
    df.columns = df.columns.str.upper()
    print(df)
    # 全部小写
    df.columns = df.columns.str.lower()
    print(df)
    # 首字母大写
    df.columns = df.columns.str.title()
    print(df)


def testFindNaN():
    data = {'Chinese': [66, 95, 93, 90, 80], 'English': [65, 85, 92, 88, np.NaN], 'Math': [100, 99, 98, 97, 96]}
    df = DataFrame(data, index=['ZhangFei', 'GuanYu', 'ZhaoYun', 'HuangZhong', 'DianWei'])
    print(df.isnull())
    print(df.isnull().any())  # 看哪些列存在空值


def testApply():
    data = {'Chinese': [66, 95, 93, 90, 80], 'English': [65, 85, 92, 88, np.NaN], 'Math': [100, 99, 98, 97, 96]}
    df = DataFrame(data, index=['ZhangFei', 'GuanYu', 'ZhaoYun', 'HuangZhong', 'DianWei'])

    def double_df(x):
        return 2 * x

    df['Chinese'] = df['Chinese'].apply(double_df)
    print(df)

    def plus(df, n, m):
        df['new1'] = (df['Math'] + df['English']) * m
        df['new2'] = (df['Math'] + df['English']) * n
        return df

    df = df.apply(plus, axis=1, args=(2, 3,))
    print(df)


def testStat():
    data = {'Chinese': [66, 95, 93, np.NaN, 89], 'English': [65, 85, 92, 88, np.NaN], 'Math': [100, 99, 98, 97, 96]}
    df = DataFrame(data, index=['ZhangFei', 'GuanYu', 'ZhaoYun', 'HuangZhong', 'DianWei'])
    print(df.count())
    print(df.describe())
    print(df.min(), df.max(), df.sum(), df.mean(), df.median(), df.var(), df.std())
    ser = df['Chinese']
    print(ser.argmin(), ser.argmax(), ser.idxmin(), ser.idxmax())  # 前两个求索引位置 后两个求索引值


def testMerge():
    df1 = DataFrame({'name': ['ZhangFei', 'GuanYu', 'a', 'b', 'c'], 'data1': range(5)})
    df2 = DataFrame({'name': ['ZhangFei', 'GuanYu', 'A', 'B', 'C'], 'data2': range(5)})
    df3 = pd.merge(df1, df2, on='name')  # 默认内连接 基于名字这一列合并 也就是取键的交集
    print(df1)
    print(df2)
    print(df3)


def testLeftRightJoin():
    df1 = DataFrame({'name': ['ZhangFei', 'GuanYu', 'a', 'b', 'c'], 'data1': range(5)})
    df2 = DataFrame({'name': ['ZhangFei', 'GuanYu', 'A', 'B', 'C'], 'data2': range(5)})
    df3 = pd.merge(df1, df2, how='left')  # 以左边为主 右边为辅的合并
    df4 = pd.merge(df1, df2, how='right')
    print(df3)
    print(df4)


def testOuterJoin():
    df1 = DataFrame({'name': ['ZhangFei', 'GuanYu', 'a', 'b', 'c'], 'data1': range(5)})
    df2 = DataFrame({'name': ['ZhangFei', 'GuanYu', 'A', 'B', 'C'], 'data2': range(5)})
    df3 = pd.merge(df1, df2, how='outer')  # 相当于取交集
    print(df3)


def testPandaSql():
    global df1
    df1 = DataFrame({'name': ['ZhangFei', 'GuanYu', 'a', 'b', 'c'], 'data1': range(5)})
    query = """select * from df1 where name ='ZhangFei'"""
    print(sqldf(query))

    # df1 = DataFrame({'name': ['ZhangFei', 'GuanYu', 'a', 'b', 'c'], 'data1': range(5)})
    # pysqldf = lambda sql: sqldf(sql, globals())
    # sql = "select * from df1 where name ='ZhangFei'"
    # print(pysqldf(sql))

def exec():
    data = {'Chinese': [66, 95, 95, 90, 80, 80], 'English': [65, 85, 92, 88, 90, 90], 'Math': [np.NaN, 98, 96, 77, 90, 90]}
    df = DataFrame(data, index=['ZhangFei', 'GuanYu', 'ZhaoYun', 'HuangZhong', 'DianWei', 'DianWei'])
    df = df.drop_duplicates()
    average = df['Math'].mean()
    df.loc['ZhangFei', 'Math'] = average
    df['total'] = df['Chinese'] + df['English'] + df['Math']
    print(df)

if __name__ == '__main__':
    # testCreateSeries()
    # testCreateDataframe()
    # testImportAndExportData()
    # testDeletePortion()
    # testRenameColumn()
    # testRemoveDuplicateRow()
    # testChangeFormat()
    # testRemoveSpaces()
    # testChangeCase()
    # testFindNaN()
    # testApply()
    # testStat()
    # testMerge()
    # testLeftRightJoin()
    # testOuterJoin()
    # testPandaSql()
    exec()