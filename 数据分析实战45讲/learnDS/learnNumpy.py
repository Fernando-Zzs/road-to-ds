# -*- coding = utf-8 -*-
# @Time : 2023/1/25 16:39
# @Author : Fernando
# @File : learnNumpy.py
# @Software : PyCharm

import numpy as np


def testArray():
    a = np.array([1, 2, 3])
    b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b[1, 1] = 10
    print(a.shape)
    print(b.shape)
    print(a.dtype)
    print(b)


def testDtype():
    persontype = np.dtype({
        'names': ['name', 'age', 'chinese', 'math', 'english'],
        'formats': ['S32', 'i', 'i', 'i', 'f']})
    peoples = np.array([("ZhangFei", 32, 75, 100, 90), ("GuanYu", 24, 85, 96, 88.5),
                        ("ZhaoYun", 28, 85, 92, 96.5), ("HuangZhong", 29, 65, 85, 100)],
                       dtype=persontype)
    ages = peoples[:]['age']
    chineses = peoples[:]['chinese']
    maths = peoples[:]['math']
    englishs = peoples[:]['english']
    print(np.mean(ages))
    print(np.mean(chineses))
    print(np.mean(maths))
    print(np.mean(englishs))


def testSpecialArray():
    x1 = np.arange(1, 11, 2)  # 初始值, 终止值, 步长
    x2 = np.linspace(1, 9, 5)  # 初始值, 终止值, 元素个数
    print(x1)
    print(x2)
    print(np.add(x1, x2))
    print(np.subtract(x1, x2))
    print(np.multiply(x1, x2))
    print(np.divide(x1, x2))
    print(np.power(x1, x2))
    print(np.remainder(x1, x2))
    print(np.mod(x1, x2))


def testStat():
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(np.amin(a))
    print(np.amin(a, 0))  # 维度为0的几个元素中取最小的
    print(np.amin(a, 1))
    print(np.amax(a))
    print(np.amax(a, 0))
    print(np.amax(a, 1))
    # 计算极差
    print(np.ptp(a))
    print(np.ptp(a, 0))
    print(np.ptp(a, 1))
    # 计算百分位数
    print(np.percentile(a, 50))
    print(np.percentile(a, 50, axis=0))
    print(np.percentile(a, 50, axis=1))
    # 求中位数
    print(np.median(a))
    print(np.median(a, axis=0))
    print(np.median(a, axis=1))
    # 求平均数
    print(np.mean(a))
    print(np.mean(a, axis=0))
    print(np.mean(a, axis=1))
    # 求加权平均值
    b = np.array([1, 2, 3, 4])
    wts = np.array([1, 2, 3, 4])
    print(np.average(b))
    print(np.average(b, weights=wts))
    # 求标准差和方差
    print(np.std(b))
    print(np.var(b))


def testSort():
    a = np.array([[4, 3, 2], [2, 4, 1]])
    print(np.sort(a))  # 默认axis为-1, 即最后一维
    print(np.sort(a, axis=None))  # 扁平化后排序
    print(np.sort(a, axis=0))  # 跨行
    print(np.sort(a, axis=1))  # 跨列


def exec():
    persontype = np.dtype({
        'names': ['name', 'chinese', 'english', 'math', 'total'],
        'formats': ['S32', 'i', 'i', 'i', 'i']})
    students = np.array([("ZhangFei", 66, 65, 30, 0), ("GuanYu", 95, 85, 98, 0),
                         ("ZhaoYun", 93, 92, 96, 0), ("HuangZhong", 90, 88, 77, 0),
                         ('DianWei', 80, 90, 90, 0)], dtype=persontype)
    chinese = students[:]['chinese']
    english = students[:]['english']
    math = students[:]['math']
    print('语文成绩的平均分：%d' % np.average(chinese))
    print('英语成绩的平均分：%d' % np.average(english))
    print('数学成绩的平均分：%d' % np.average(math))
    print('语文成绩的最小值：%d' % np.amin(chinese))
    print('英语成绩的最小值：%d' % np.amin(english))
    print('数学成绩的最小值：%d' % np.amin(math))
    print('语文成绩的最大值：%d' % np.amax(chinese))
    print('英语成绩的最大值：%d' % np.amax(english))
    print('数学成绩的最大值：%d' % np.amax(math))
    print('语文成绩的方差：%d' % np.var(chinese))
    print('英语成绩的方差：%d' % np.var(english))
    print('数学成绩的方差：%d' % np.var(math))
    print('语文成绩的标准差：%d' % np.std(chinese))
    print('英语成绩的标准差：%d' % np.std(english))
    print('数学成绩的标准差：%d' % np.std(math))
    students[:]['total'] = students[:]['chinese'] + students[:]['english'] + students[:]['math']
    print(np.sort(students, order='total'))


if __name__ == '__main__':
    # testArray()
    # testDtype()
    # testSpecialArray()
    # testStat()
    # testSort()
    exec()
