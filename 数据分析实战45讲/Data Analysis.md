# 课时1

## 数据分析的全景图

- 数据采集
  - 数据源
    - 爬虫
    - 日志采集
    - 传感器
  - 工具使用
    - 八爪鱼
    - 火车采集器
    - 搜集客
  - 爬虫编写
    - phantomjs
    - Scarp
    - lxml
    - Selenium
- 数据挖掘
  - 数学基础
    - 概率论与数据统计
    - 线性代数
    - 图论
    - 最优化方法
  - 基本流程
    - 商业理解
    - 数据理解
    - 数据准备
    - 模型建立
    - 模型评估
    - 上线发布
  - 十大算法
    - 分类算法
      - C45
      - 朴素贝叶斯
      - SVM
      - KNN
      - Adaboost
      - CART
    - 聚类算法
      - K-Means
      - EM
    - 关联分析
      - Apriori
    - 连接分析
      - PageRank
  - 实战
    - 如何识别手写字
    - 如何进行乳腺癌症检测
    - 如何对文档进行归类
    - ...
- 数据可视化
  - Python 数据清洗 挖掘
    - matplotlib
    - Seaborn
  - 第三方工具
    - 微图
    - DataV
    - Data GIF Maker

## 牢记原则

1. 不重复造轮子
2. 工具决定效率
3. 熟练度

# 课时2

## 数据挖掘的基本流程

数据挖掘的过程可以分成以下 6 个步骤。

1. 商业理解：数据挖掘不是我们的目的，我们的目的是更好地帮助业务，所以第一步我们
    要从商业的角度理解项目需求，在这个基础上，再对数据挖掘的目标进行定义。
2. 数据理解：尝试收集部分数据，然后对数据进行探索，包括数据描述、数据质量验证
    等。这有助于你对收集的数据有个初步的认知。
3. 数据准备：开始收集数据，并对数据进行清洗、数据集成等操作，完成数据挖掘前的准
    备工作。
4. 模型建立：选择和应用各种数据挖掘模型，并进行优化，以便得到更好的分类结果。
5. 模型评估：对模型进行评价，并检查构建模型的每个步骤，确认模型是否实现了预定的
    商业目标。
6. 上线发布：模型的作用是从数据中找到金矿，也就是我们所说的“知识”，获得的知识
    需要转化成用户可以使用的方式，呈现的形式可以是一份报告，也可以是实现一个比较
    复杂的、可重复的数据挖掘过程。数据挖掘结果如果是日常运营的一部分，那么后续的
    监控和维护就会变得重要。

## 数据挖掘的十大算法

- 分类算法：C4.5，朴素贝叶斯（Naive Bayes），SVM，KNN，Adaboost，CART
-  聚类算法：K-Means，EM
- 关联分析：Apriori
- 连接分析：PageRank

### C4.5

C4.5 是决策树的算法，它创造性地在决策树构造过程中就进行了剪枝，并且可以处理连续的属性，也能对不完整的数据进行处理。它可以说是决策树分类中，具有里程碑式意义的算法。

### 朴素贝叶斯（Naive Bayes）

朴素贝叶斯模型是基于概率论的原理，它的思想是这样的：对于给出的未知物体想要进行分类，就需要求解在这个未知物体出现的条件下各个类别出现的概率，哪个最大，就认为这个未知物体属于哪个分类。

### SVM

SVM 的中文叫支持向量机，英文是 Support Vector Machine，简称 SVM。SVM 在训练中建立了一个超平面的分类模型。

### KNN

KNN 也叫 K 最近邻算法，英文是 K-Nearest Neighbor。所谓 K 近邻，就是每个样本都可以用它最接近的 K 个邻居来代表。如果一个样本，它的 K 个最接近的邻居都属于分类 A，那么这个样本也属于分类 A。

### AdaBoost

Adaboost 在训练中建立了一个联合的分类模型。boost 在英文中代表提升的意思，所以Adaboost 是个构建分类器的提升算法。它可以让我们多个弱的分类器组成一个强的分类器，所以 Adaboost 也是一个常用的分类算法。

### CART

CART 代表分类和回归树，英文是 Classification and Regression Trees。像英文一样，它构建了两棵树：一棵是分类树，另一个是回归树。和 C4.5 一样，它是一个决策树学习方法。

### Apriori

Apriori 是一种挖掘关联规则（association rules）的算法，它通过挖掘频繁项集（frequent item sets）来揭示物品之间的关联关系，被广泛应用到商业挖掘和网络安全等领域中。频繁项集是指经常出现在一起的物品的集合，关联规则暗示着两种物品之间可能存在很强的关系。

### K-Means

K-Means 算法是一个聚类算法。你可以这么理解，最终我想把物体划分成 K 类。假设每个类别里面，都有个“中心点”，即意见领袖，它是这个类别的核心。现在我有一个新点要归类，这时候就只要计算这个新点与 K 个中心点的距离，距离哪个中心点近，就变成了哪个类别。

### EM

EM 算法也叫最大期望算法，是求参数的最大似然估计的一种方法。原理是这样的：假设我们想要评估参数 A 和参数 B，在开始状态下二者都是未知的，并且知道了 A 的信息就可以得到 B 的信息，反过来知道了 B 也就得到了 A。可以考虑首先赋予 A 某个初值，以此得到B 的估值，然后从 B 的估值出发，重新估计 A 的取值，这个过程一直持续到收敛为止。
EM 算法经常用于聚类和机器学习领域中。

### PageRank

PageRank 起源于论文影响力的计算方式，如果一篇文论被引入的次数越多，就代表这篇论文的影响力越强。同样PageRank 被 Google 创造性地应用到了网页权重的计算中：当一个页面链出的页面越多，说明这个页面的“参考文献”越多，当这个页面被链入的频率越高，说明这个页面被引用的次数越高。基于这个原理，我们可以得到网站的权重划分。

## 数据挖掘的数学原理

### 概率论与数理统计

在数据挖掘里使用到概率论的地方就比较多了。比如条件概率、独立性的概念，以及随机变量、多维随机变量的概念。很多算法的本质都与概率论相关，所以说概率论与数理统计是数据挖掘的重要数学基础。

### 线性代数

向量和矩阵是线性代数中的重要知识点，它被广泛应用到数据挖掘中，比如我们经常会把对象抽象为矩阵的表示，一幅图像就可以抽象出来是一个矩阵，我们也经常计算特征值和特征向量，用特征向量来近似代表物体的特征。这个是大数据降维的基本思路。基于矩阵的各种运算，以及基于矩阵的理论成熟，可以帮我们解决很多实际问题，比如PCA 方法、SVD 方法，以及 MF、NMF 方法等在数据挖掘中都有广泛的应用。

### 图论

社交网络的兴起，让图论的应用也越来越广。人与人的关系，可以用图论上的两个节点来进行连接，节点的度可以理解为一个人的朋友数。我们都听说过人脉的六度理论，在Facebook 上被证明平均一个人与另一个人的连接，只需要 3.57 个人。当然图论对于网络结构的分析非常有效，同时图论也在关系挖掘和图像分割中有重要的作用。

### 最优化方法

最优化方法相当于机器学习中自我学习的过程，当机器知道了目标，训练后与结果存在偏差就需要迭代调整，那么最优化就是这个调整的过程。一般来说，这个学习和迭代的过程是漫长、随机的。最优化方法的提出就是用更短的时间得到收敛，取得更好的效果。

# 课时3

## 基本数据结构

### 列表：[]

列表是 Python 中常用的数据结构，相当于数组，具有增删改查的功能，我们可以使用len() 函数获得 lists 中元素的个数；使用 append() 在尾部添加元素，使用 insert() 在列表中插入元素，使用 pop() 删除尾部的元素。

### 元组：()

元组 tuple 和 list 非常类似，但是 tuple 一旦初始化就不能修改。因为不能修改所以没有append(), insert() 这样的方法，可以像访问数组一样进行访问，比如 tuples[0]，但不能赋值。

### 字典 : {}

字典其实就是{key, value}，多次对同一个 key 放入 value，后面的值会把前面的值冲掉，同样字典也有增删改查。增加字典的元素相当于赋值，比如 score[‘zhaoyun’] = 98，删除一个元素使用 pop，查询使用 get，如果查询的值不存在，我们也可以给一个默认值，比如 score.get(‘yase’,99)。

### 集合：set([])

集合 set 和字典 dictory 类似，不过它只是 key 的集合，不存储 value。同样可以增删查，增加使用 add，删除使用 remove，查询看某个元素是否在这个集合里，使用 in。

# 课时4

## NumPy数组的优势

- 列表 list 的元素在系统内存中是分散存储的，而 NumPy 数组存储在一个均匀连续的内存块中，数组计算遍历所有的元素，不像列表 list 还需要对内存地址进行查找，从而节省了计算资源。
- 在内存访问模式中，缓存会直接把字节块从 RAM 加载到 CPU 寄存器中。因为数据连续的存储在内存中，NumPy 直接利用现代 CPU 的矢量化指令计算，加载寄存器中的多个连续浮点数。
- NumPy 中的矩阵计算可以采用多线程的方式，充分利用多核 CPU 计算资源，大大提升了计算效率。
- 避免采用隐式拷贝，而是采用就地操作的方式。举个例子，如果我想让一个数值 x 是原来的两倍，可以直接写成 x\*=2，而不要写成 y=x\*2。

## NumPy的对象

### ndarray（N-dimensional array object）

NumPy 数组中，维数称为秩（rank），每一个线性的数组称为一个轴（axes），其实秩就是描述轴的数量。

```python
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
```

### ufunc（universal function object）

它能对数组中每个元素进行函数操作。NumPy 中很多 ufunc 函数计算速度非常快，因为都是采用 C 语言实现的。因为在对多维进行计算的时候需要进行循环，c语言的循环比python的循环效率要高。

包括三角函数、四则运算、比较运算、布尔运算、自定义构建

## 结构数组dtype

在 C 语言里，可以定义结构数组，也就是通过 struct 定义结构类型，结构中的字段占据连续的内存空间，每个结构体占用的内存大小都相同，在NumPy中定义的方式如下：

```python
persontype = np.dtype({
 'names':['name', 'age', 'chinese', 'math', 'english'],
 'formats':['S32','i', 'i', 'i', 'f']})
peoples = np.array([("ZhangFei",32,75,100, 90),("GuanYu",24,85,96,88.5),
 ("ZhaoYun",28,85,92,96.5),("HuangZhong",29,65,85,100)],
dtype=persontype)
```



## 连续数组的创建

```python
x1 = np.arange(1,11,2) # 初始值 终止值（闭区间） 步长
x2 = np.linspace(1,9,5) # 初始值 终止值（开区间） 元素个数 
```

## 统计函数

对于一个ndarray，想要知道这些数据中的最大值、最小值、平均值，是否符合正态分布，方差、标准差多少等等信息。统计函数可以让你更清楚地对这组数据有认知。

- amin
- amax
- ptp
- percentile
- median
- mean
- average
- std
- var

## NumPy排序

```python
np.sort(a, axis=-1, kind=‘quicksort’, order=None)
```

使用 sort 函数，默认情况下使用的是快速排序；在 kind 里，可以指定 quicksort、mergesort、heapsort 分别表示快速排序、合并排序、堆排序。同样 axis 默认是 -1，即沿着数组的最后一个轴进行排序，也可以取不同的 axis 轴，或者axis=None 代表采用扁平化的方式作为一个向量进行排序。另外order 字段，对于结构化的数组可以指定按照某个字段进行排序。

# 课时5

## Pandas的两大数据结构

### Series

直观来看，Series是数据表的一列。官方说法，Series是一个定长的字典序列，因为在存储的时候，相当于两个 ndarray，而字典的结构里，元素的个数是不固定的，这也是和字典结构最大的不同。

Series有两个基本属性，index和values（index默认是0，1，2……如果指定索引名就为指定的名字）

### DataFrame

它包括了行索引和列索引，我们可以将 DataFrame 看成是由相同索引的 Series 组成的字典类型。

## 常见数据清洗函数

<img src=".\images\image-20230126204020024.png" alt="image-20230126204020024" style="zoom: 80%;" />

```python
# 数据导入导出
score = DataFrame(pd.read_excel('data.xlsx')) # 导入
score.to_excel('data1.xlsx') # 导出

# 数据清洗
df2 = df2.drop(columns=['Chinese']) # 删除指定列
df2 = df2.drop(index=['ZhangFei']) # 删除指定行
df2.rename(columns={'Chinese': 'YuWen', 'English': 'Yingyu'}, inplace = True) # 重命名
df = df.drop_duplicates() # 去除重复行
df2['Chinese'].astype('str') # 换数据类型
df2['Chinese']=df2['Chinese'].map(str.strip) # 删除左右两边空格
df2['Chinese']=df2['Chinese'].map(str.lstrip) # 删除左边空格
df2['Chinese']=df2['Chinese'].map(str.rstrip) # 删除右边空格
df2['Chinese']=df2['Chinese'].str.strip('$') # 删除指定字符
df2.columns = df2.columns.str.upper() # 全部大写
df2.columns = df2.columns.str.lower() # 全部小写
df2.columns = df2.columns.str.title() # 首字母大写
df.isnull().any() # 查看哪些列有空值

# apply函数
def plus(df,n,m):
 df['new1'] = (df[u'语文']+df[u'英语']) * m
 df['new2'] = (df[u'语文']+df[u'英语']) * n
 return df
df1 = df1.apply(plus,axis=1,args=(2,3,))
'''
其中 axis=1 代表按照列为轴进行操作，axis=0 代表按照行为轴进行操作，args 是传递的
两个参数，即 n=2, m=3，在 plus 函数中使用到了 n 和 m，从而生成新的 df。
'''
```

## 常用统计函数

1. `count()` 统计个数 NaN 和 None 除外
2. `describe()` 一次性输出多个统计指标
3. `min()` 最小值
4. `max()` 最大值
5. `sum()` 求和
6. `mean()` 平均值
7. `median()` 中位数
8. `var()` 方差
9. `std()` 标准差
10. `argmin()` 统计最小值的索引位置
11. `argmax()` 统计最大值的索引位置
12. `idxmin()` 统计最小值的索引值
13. `idxmax()` 统计最大值的索引值

## 合并

```python
df3 = pd.merge(df1, df2, on='name') # 基于指定列合并
df3 = pd.merge(df1, df2, how='inner') # 内连接
df3 = pd.merge(df1, df2, how='left') # 左连接
df3 = pd.merge(df1, df2, how='right') # 右连接
df3 = pd.merge(df1, df2, how='outer') # 外连接
```

## PandaSql

```python
# 单个全局化
global df1
df1 = DataFrame({'name': ['ZhangFei', 'GuanYu', 'a', 'b', 'c'], 'data1': range(5)})
query = """select * from df1 where name ='ZhangFei'"""
print(sqldf(query))

# 一起全局化
df1 = DataFrame({'name': ['ZhangFei', 'GuanYu', 'a', 'b', 'c'], 'data1': range(5)})
pysqldf = lambda sql: sqldf(sql, globals())
sql = "select * from df1 where name ='ZhangFei'"
print(pysqldf(sql))

# lambda argument_list: expression
```

# 补充知识

## DataFrame切片

```python
# 创建DataFrame
df = pd.DataFrame(np.arange(12, 60).reshape((12, 4)), columns=["WW", "XX", "YY", "ZZ"])
'''
    WW  XX  YY  ZZ
0   12  13  14  15
1   16  17  18  19
2   20  21  22  23
3   24  25  26  27
4   28  29  30  31
5   32  33  34  35
6   36  37  38  39
7   40  41  42  43
8   44  45  46  47
9   48  49  50  51
10  52  53  54  55
11  56  57  58  59
'''

# 取行：方括号里写数字
print(df[:5])  # 取0-4行
print(df[2:6]) # 取2-5行

# 取列：方括号里写字符串
print(df["YY"])
print(df["XX", "ZZ"])

# 组合取
print(df[2:6]["YY"])
```

1. `df.loc`通过索引标签名取数据
2. `df.iloc`通过位置获取数据 (位置从0开始算)

`loc()`和`iloc()`并不会拷贝，只是视图上的操作，通过`loc()`或`iloc()`进行赋值修改会直接修改原先的DataFrame

```python
# 创建DataFrame
df = pd.DataFrame(np.arange(12, 32).reshape((5, 4)), index=["a","b","c","d","e"], columns=["WW","XX","YY","ZZ"])
'''
   WW  XX  YY  ZZ
a  12  13  14  15
b  16  17  18  19
c  20  21  22  23
d  24  25  26  27
e  28  29  30  31
'''
# 取一行
print(df.loc["c"])  # Series类型
# 取多行
print(df.loc["a", "d"]) # DataFrame类型
# 取连续多行
print(df.loc["b": "d"]) # 左右都是闭区间
# 取单行单列
print(df.loc["b", "YY"])
# 取单行多列
print(df.loc["b", ["XX", "ZZ"]])  # Series类型
# 取多行多列
print(df.loc[["b", "d"], ["XX", "ZZ"]])
# 取连续多行多列
print(df.loc["b": "d", ["XX", "ZZ"]])
# 取单列所有行
print(df.loc[:, ["XX"]]) # DataFrame类型
print(df.log[:, "XX"]) # Series类型
# 用iloc
print(df.iloc[1:3,[2,3]]) # 注意：切片位置从0开始算左闭右开，方括号两边闭
```

## 数据清洗理论

### 什么是整齐的数据

每一行代表一个观测值，每一列代表一个变量，每一个表格代表一个观测单元

### 混乱数据的特点

1. 列名是值而不是变量
2. 多个变量存储在一列中
3. 变量既存储在列中，也存储在行中
4. 多种类型的观测单元存储在一个表中
5. 单个观测单元存储在多个表中

### 第一类问题：列名是值而不是变量

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230127154335467.png" alt="image-20230127154335467" style="zoom: 67%;" />

上述数据属于**宽数据**，可以在较小的空间内存储非常多的信息，但数据分析师需要的是**长数据**，需要改成一列表示范围，一列是具体的值。

```python
pd.melt(df, id_vars='religion', var_name='income', value_name='count')

"""
Parameters:
id_vars: 指定哪些列不需要改变
value_vars: 指定哪些列需要分解
var_name: 在分解数据框可以改变默认的列名
value_name: 分解数据后可以重命名值列
"""
```

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230127155558738.png" alt="image-20230127155558738" style="zoom:67%;" />

### 第二类问题：多个变量存储在一列中

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230127173204509.png" alt="image-20230127173204509" style="zoom:80%;" />

如上图，病例数、死亡数和国家连在一起，我们需要分离出这些变量

```python
# 分解数据
ebola_long = pd.melt(ebola, id_vars=['Date', 'Day'])
```

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230127173455407.png" alt="image-20230127173455407" style="zoom:67%;" />

#### 方法1

```python
# 对variable列进行字符串解析成列表
variable_split = ebola_long.variable.str.split('_')
status_value = variable_split.str.get(0)
country_value = variable_split.str.get(1)
ebola_long['status'] = status_value
ebola_long['country'] = country_value
```

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230127174218944.png" alt="image-20230127174218944" style="zoom:67%;" />

#### 方法2

```python
# 对variable列进行字符串解析成DataFrame
variable_split = ebola_long.variable.str.split('_', expand=True)
# 合并两个数据框的内容
ebola_parsed = pd.concat([ebola_long, variable_split], axis=1)
```

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230127174443321.png" alt="image-20230127174443321" style="zoom:67%;" />

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230127175040382.png" alt="image-20230127175040382" style="zoom:80%;" />

#### 方法3

```python
# 合并两个列表zip
variable_split = ebola_long.variable.str.split('_')
# 将序列拆开得到两列，一次性添加到原DataFrame中
ebola_long['status_zip'], ebola_long['country_zip'] = zip(*variable_split)
```

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230127175736893.png" alt="image-20230127175736893" style="zoom:67%;" />

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230127180002222.png" alt="image-20230127180002222" style="zoom:67%;" />

### 第三类问题：变量既存储在列中，也存储在行中

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230127180140877.png" alt="image-20230127180140877" style="zoom:67%;" />

如上图，element列保存的也是变量，即最大值和最小值

```python
# 获取月中的某天并分解它
weather_melt = pd.melt(weather, id_vars=['id', 'year', 'month', 'element'], var_name='day', value_name='temp')

```

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230127204536172.png" alt="image-20230127204536172" style="zoom:67%;" />

反向操作melt命令将使用数据透视表，可以用于获取某列以及获取该列的每个独立值，并将其转换成单独的一列

在这里我们获取element列并把两个变量填充到图表的列上

```python
weather_tidy = weather_melt.pivot_table(index=['id', 'year', 'month', 'day'], columns='element', values='temp')

"""
pd.pivot_table()
Parameters:
index: 指定哪些列不需要改变
columns: 指定哪些列撤销melt命令
values: 告诉函数用什么值来填充单元格
"""
```

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230127210208897.png" alt="image-20230127210208897" style="zoom:67%;" />

```python
weather_tidy_flat = weather_tidy.reset_index()
```

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230127210558389.png" alt="image-20230127210558389" style="zoom:67%;" />

遇到第三类问题，首先需要对其中的某些列进行分解，然后对其中的某一列做数据透视表，最后为了得到一个标准的矩形数据框，要重新设置所有的索引。

### 第四类问题：多种类型的观测单元存储在一个表中

当数据集中有许多重复的信息时就是数据规范化问题，在行政类、教育类数据集中常会遇到这样的问题

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230127211054393.png" alt="image-20230127211054393" style="zoom:67%;" />

如上图，year/artist/track/time/date.entered都是曲目信息，week/rating是评分信息，需要分解成两个数据框billboard_songs和billboard_ratings

```python
billboard_songs = billboard_long[['year', 'artist', 'track', 'time']]
billboard_songs = billboard_songs.drop_duplicates() # 去掉了评分信息后很多行都是重复的

# 为每一首歌分配一个唯一标识
billboard_songs['id'] = range(len(billboard_songs))

# 与原数据集连接后再去子集
billboard_ratings = billboard_long.merge(billboard_songs, on=['year', 'artist', 'track', 'time'])
billboard_ratings = billboard_ratings[['id', 'date.entered', 'week', 'rating']]
```

### 第五类问题：单个观测单元存储在多个表中

```python
# 需要同时载入多个文件
import glob
concat_files = glob.glob('../data/concat*')

# 得到数据框的列表
list_concat_df = []
for csv_filename in concat_files:
    df = pd.read_csv(csv_filename)
    list_concat_df.append(df)
    
concat_loop = pd.concat(list_concat_df)

```

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230127215142717.png" alt="image-20230127215142717" style="zoom:67%;" />

# 课时6

## BI、DW、DM三者关系

BI是商业智能，DW是数据仓库，DM是数据挖掘。形象的比喻是，数据仓库是个金矿，数据挖掘是炼金术，商业报告是黄金。

### BI

基于数据仓库，经过数据挖掘后得到的具有商业价值的信息的过程。

### DW

BI的地基，数据仓库是数据库的升级概念，不同的地方在于数据仓库更庞大，使用与数据挖掘和数据分析，而数据库更偏向于一种存储技术。数据仓库将原有的多个数据来源中的数据进行汇总、整理而得。数据进入数据仓库前，必须消除数据中的不一致性，方便后续进行数据分析和挖掘。

### DM

核心包括分类、聚类、预测、关联分析等任务，有了这些数据挖掘技术的支持，才能指导业务层做更好的BI分析。

其他概念：ETL数据搬运工，DM也通常叫做数据科学家。

## 元数据与数据元

元数据（MetaData）：描述其它数据的数据，也称为“中介数据”。
数据元（Data Element）：就是最小数据单元。

在生活中，只要有一类事物，就可以定义一套元数据。元数据可以很方便地应用于数据仓库。比如数据仓库中有数据和数据之间的各种复杂关系，为了描述这些关系，元数据可以对数据仓库的数据进行定义，刻画数据的抽取和转换规则，存储与数据仓库主题有关的各种信息。而且整个数据仓库的运行都是基于元数据的，比如抽取调度数据、获取历史数据等。

通过元数据，可以很方便地帮助我们管理数据仓库。

## KDD的全过程

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230128203656390.png" alt="image-20230128203656390" style="zoom:67%;" />

### 数据预处理

#### 数据清洗

主要是为了去除重复数据，去噪声（即干扰数据）以及填充缺失值。

#### 数据集成

是将多个数据源中的数据存放在一个统一的数据存储中。

#### 数据变化

就是将数据转换成适合数据挖掘的形式。比如，通过归一化将属性数据按照比例缩放，这样
就可以将数值落入一个特定的区间内，比如 0~1 之间。

### 数据后处理

是将模型预测的结果进一步处理后，再导出。比如在二分类问题中，一般能得到
的是 0~1 之间的概率值，此时把数据以 0.5 为界限进行四舍五入就可以实现后处理。

# 课时7

## 用户画像建模

### 统一化

用户画像的核心就是用户唯一标识，它把“从用户开始使用 APP到下单到售后整个所有的用户行为”进行串联，这样就可以更好地去跟踪和分析一个用户的特征。

### 标签化

为了保证用户画像的全面性，设计标签的维度共有四个：**用户消费行为分析**。

#### 用户标签

性别、年龄、地域、收入、学历、职业等

#### 消费标签

消费习惯、购买意向、是否对促销敏感

#### 行为标签

时间段、频次、时长、访问路径

#### 内容分析

对用户平时浏览的内容，尤其是停留时间长、浏览次数多的内容进行分析，分析出用户对哪些内容感兴趣

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230128213928477.png" alt="image-20230128213928477"  />

**数据层**指的是用户消费行为里的标签。我们可以打上“事实标签”，作为数据客观的记录。
**算法层**指的是透过这些行为算出的用户建模。我们可以打上“模型标签”，作为用户画像的分类标识。
**业务层**指的是获客、粘客、留客的手段。我们可以打上“预测标签”，作为业务关联的结果。

### 业务化

获得准确全面的用户画像是产品获客（如何进行拉新，通过更精准的营销获取客户）、粘客（个性化推荐，搜索排序，场景运营等）、留客（流失率预测，分析关键节点降低流失率）的保障。

## 案例：美团用户画像

### 第一步：用户唯一标识的统一化

美团已经和大众点评进行了合并，因此在大众点评和美团外卖上都可以进行外卖下单。我们看到，美团采用的是手机号、微信、微博、美团账号的登录方式。大众点评采用的是手机号、微信、QQ、微博的登录方式。这里面两个 APP 共同的登录方式都是手机号、微信和微博。那么究竟哪个可以作为用户的唯一标识呢？当然主要是以用户的注册手机号为标准。这样美团和大众点评的账号体系就可以相通。

在集团内部，各部门之间的协作，尤其是用户数据打通是非常困难的，所以这里建议，如果希望大数据对各个部门都能赋能，一定要在集团的战略高度上，尽早就在最开始的顶层架构上，将用户标识进行统一，这样在后续过程中才能实现用户数据的打通。

### 第二步：标签化打造用户画像

1. 用户标签：性别、年龄、家乡、居住地、收货地址、婚姻、宝宝信息、通过何种渠道进
行的注册。
2. 消费标签：餐饮口味、消费均价、团购等级、预定使用等级、排队使用等级、外卖等
级。
3. 行为标签：点外卖时间段、使用频次、平均点餐用时、访问路径。
4. 内容分析：基于用户平时浏览的内容进行统计，包括餐饮口味、优惠敏感度等。

### 第三步：结合具体业务生成方案

在获客上，我们可以找到优势的宣传渠道，如何通过个性化的宣传手段，吸引有潜在需求的用户，并刺激其转化。
在粘客上，如何提升用户的单价和消费频次，方法可以包括购买后的个性化推荐、针对优质用户进行优质高价商品的推荐、以及重复购买，比如通过红包、优惠等方式激励对优惠敏感的人群，提升购买频次。
在留客上，预测用户是否可能会从平台上流失。在营销领域，关于用户留存有一个观点——如果将顾客流失率降低 5%，公司利润将提升 25%~85%。可以看出留存率是多么的重要。用户流失可能会包括多种情况，比如用户体验、竞争对手、需求变化等，通过预测用户的流失率可以大幅降低用户留存的运营成本。

# 课时8

## 数据源有哪些

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230128220257975.png" alt="image-20230128220257975" style="zoom: 50%;" />

## 爬虫抓取的方法

在 Python 爬虫中，基本上会经历三个过程。
1. 使用 Requests 爬取内容。我们可以使用 Requests 库来抓取网页信息。Requests 库可以说是 Python 爬虫的利器，也就是 Python 的 HTTP 库，通过这个库爬取网页中的数据，非常方便，可以帮我们节约大量的时间。
2. 使用 XPath 解析内容。XPath 是 XML Path 的缩写，也就是 XML 路径语言。它是一种用来确定 XML 文档中某部分位置的语言，在开发中经常用来当作小型查询语言。XPath可以通过元素和属性进行位置索引。
3. 使用 Pandas 保存数据。Pandas 是让数据分析工作变得更加简单的高级数据结构，我们可以用 Pandas 保存爬取的数据。最后通过 Pandas 再写入到 XLS 或者 MySQL 等数据库中。

Requests、XPath、Pandas 是 Python 的三个利器。当然做 Python 爬虫还有很多利器，比如 Selenium，PhantomJS，或者用 Puppteteer 这种无头模式。

第三方工具中，比较推荐使用八爪鱼，拥有云采集功能。八爪鱼一共有 5000 台服务器，通过云端多节点并发采集，采集速度远远超过本地采集。此外还可以自动切换多个 IP，避免 IP 被封，影响采集。

## 日志采集

日志采集是运维人员的工作之一。日志就是日记的意思，它记录了用户访问网站的全过程：哪些人在什么时间，通过什么渠道（比如搜索引擎、网址输入）来过，都执行了哪些操作；系统是否产生了错误；甚至包括用户的 IP、HTTP 请求的时间，用户代理等。这些日志数据可以被写在一个日志文件中，也可以分成不同的日志文件，比如访问日志、错误日志等。

### 日志采集作用

日志采集最大的作用，就是通过分析用户访问情况，提升系统的性能，从而提高系统承载量。及时发现系统承载瓶颈，也可以方便技术人员基于用户实际的访问情况进行优化。

### 日志采集分类

1. 通过 Web 服务器采集，例如 httpd、Nginx、Tomcat 都自带日志记录功能。同时很多互联网企业都有自己的海量数据采集工具，多用于系统日志采集，如 Hadoop 的Chukwa、Cloudera 的 Flume、Facebook 的 Scribe 等，这些工具均采用分布式架构，能够满足每秒数百 MB 的日志数据采集和传输需求。
2. 自定义采集用户行为，例如用 JavaScript 代码监听用户的行为、AJAX 异步请求后台记录日志等。

### 埋点

在有需要的位置采集相应的信息，进行上报。比如某页面的访问情况，包括用户信
息、设备信息；或者用户在页面上的操作行为，包括时间长短等。这就是埋点，每一个埋点就像一台摄像头，采集用户行为数据，将数据进行多维度的交叉分析，可真实还原出用户使用场景，和用户使用需求。

埋点就是在你需要统计数据的地方植入统计代码，当然植入代码可以自己写，也可以使用第三方统计工具。

这里推荐你使用第三方的工具，比如友盟、Google Analysis、Talkingdata 等。他们都是采用前端埋点的方式，然后在第三方工具里就可以看到用户的行为数据。但如果我们想要看到更深层的用户操作行为，就需要进行自定义埋点。