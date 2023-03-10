# -*- coding = utf-8 -*-
# @Time : 2023/2/5 15:03
# @Author : Fernando
# @File : directorAssociation.py
# @Software : PyCharm
from efficient_apriori import apriori
import csv
director = u'宁浩'
file_name = './output/'+director+'.csv'
lists = csv.reader(open(file_name, 'r', encoding='utf-8-sig'))

def findAsso():
    # 数据加载
    data = []
    for names in lists:
        name_new = []
        for name in names:
            # 去掉演员数据中的空格
            name_new.append(name.strip())
        data.append(name_new[1:])
    # 挖掘频繁项集和关联规则
    itemsets, rules = apriori(data, min_support=0.3, min_confidence=1)
    print(data)
    print(itemsets)
    print(rules)

if __name__ == "__main__":
    findAsso()