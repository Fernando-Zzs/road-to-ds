# -*- coding = utf-8 -*-
# @Time : 2023/2/5 14:07
# @Author : Fernando
# @File : apriori.py
# @Software : PyCharm

from efficient_apriori import apriori

if __name__ == "__main__":
    # 设置数据集
    data = [('牛奶', '面包', '尿布'),
            ('可乐', '面包', '尿布', '啤酒'),
            ('牛奶', '尿布', '啤酒', '鸡蛋'),
            ('面包', '牛奶', '尿布', '啤酒'),
            ('面包', '牛奶', '尿布', '可乐')]
    # 挖掘频繁项集和频繁规则
    itemsets, rules = apriori(data, min_support=0.5, min_confidence=1)
    print(itemsets)
    print(rules)