## 基础理论篇

### TensorFlow2的设计思想

TensorFlow2.0的核心在于简单、强大、可移植性强，它简化了TF1.0中的很多概念，同时兼容了很多其他库，并且完善了周边一系列的生态项目

TF2.0兼容了python语法，numpy的高维数组、原生keras的分布式、pytorch的Eager Execution（动态图引擎）等优势

动态图特性支持算法工程师动态地编写和调试模型

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230116204542856.png" alt="image-20230116204542856" style="zoom:80%;" />

TF Lite用于终端，TF.js用于网页端，TF Quantum用于量化（减小模型规模），TF Extended是TFX端到端解决方案，TF Hub可用于复用预训练模型

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230116204559315.png" alt="image-20230116204559315" style="zoom:80%;" />

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230116205610888.png" alt="image-20230116205610888" style="zoom: 50%;" />

### TensorFlow的核心模块

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230116205712459.png" alt="image-20230116205712459" style="zoom:67%;" />

以上模块中，tf.keras、tf.data、tf.distribute和SavedModel是关键

#### tf.keras：分布式与高性能

+ 构建和训练模型的高层次API
+ API完全兼容原生keras
+ 支持保存和加载SavedModel
+ 支持Eager Execution
+ 支持分布式训练

#### tf.data：功能强大的数据管理模块

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230116210247405.png" alt="image-20230116210247405" style="zoom:50%;" />

其中，分布式文件系统指的是类似于HDFS的系统，对象存储系统是公有云服务商提供的obs、oss等服务；Python生成器是指大量数据情况下用于流水线性输入数据的模块，TFRecord是一种高性能的数据格式；Shuffle指数据的打乱，py_function是指自定义的python函数，重采样是指数据集不平衡时重新采样的操作

#### tf.distribute：一行实现分布式

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230116210838870.png" alt="image-20230116210838870" style="zoom: 50%;" />

通常选取的策略是MirroredStrategy，一个模型，多个机器并行输入数据进行训练，计算梯度并经过统一汇合收敛

#### SavedModel：生产级的模型格式

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230116211028523.png" alt="image-20230116211028523" style="zoom:50%;" />

###  TensorFlow2  vs   TensorFlow1.x

TensorFlow2在易用性和灵活性上达到平衡

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230116211206319.png" alt="image-20230116211206319" style="zoom:67%;" />

工作流图的对比

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230116211430060.png" alt="image-20230116211430060" style="zoom:67%;" />

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230116211451598.png" alt="image-20230116211451598" style="zoom: 50%;" />

### TensorFlow生产级AI方案

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230116211626860.png" alt="image-20230116211626860" style="zoom:50%;" />

### TensorFlow在企业中的实际案例

[腾讯互娱角色识别任务的分布式训练框架](https://mp.weixin.qq.com/s/LRxyvVazRAOR_B0as7ujvg)

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230116211932105.png" alt="image-20230116211932105" style="zoom: 50%;" />

[贝壳找房的推荐算法](https://mp.weixin.qq.com/s/B29wlVM4E3efxCywCy8Xnw)

[QQ音乐的AI赋能曲库](https://mp.weixin.qq.com/s/1ENIla2CUU1dfhJ-6_bDXg)



## 快速上手篇

### 环境

确保python版本为3.5-3.7

```shell
pip install -i https://mirrors.aliyun.com/pypi/simple --upgrade pip

pip install -i https://mirrors.aliyun.com/pypi/simple tensorflow==2.2.0

pip install -i https://mirrors.aliyun.com/pypi/simple jupyterlab
```

### 使用JupyterLab启动TensorFlow2

```shell
jupyter lab --config jupyter_config.json
```

配置文件可以设置好后各处通用（服务器、本地）

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230117100026848.png" alt="image-20230117100026848" style="zoom: 80%;" />

* ip是指任何客户端IP都能连接
* open_browser: 本地时使用true，服务器时使用false
* allow_root: 是否允许root用户的权限打开，容器时用true，裸机用false

### Docker容器与虚拟机

虚拟机是更重的虚拟化技术，可以完整地模拟一台机器从驱动开始到整个上层的App

容器在上层没有什么操作系统的依赖，应用打包后直接可以在容器上运行（免去了GUEST OS和HYPERVISOR这些很重的概念），通过复用底层操作系统本身的OS和DOCKER DAEMON复用操作系统的很多资源

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230117101531211.png" alt="image-20230117101531211" style="zoom:67%;" />

### data模块

#### tf.keras.datasets

属于内置的数据集模块，包含了mnist、fashion_mnist、cifar10等预置数据集

#### tf.data.Dataset

tf.data.Dataset.from_tensor_slices加载list

tf.data.Dataset.from_generator加载Generator

tf.data.TextLineDataset加载文本

### keras模块

#### keras.Model

TensorFlow2.2之后，所有的Model用法都统一为此种，可以构建、训练、保存和加载模型

##### 构建

```python
model = tf.keras.models.Sequential([
    ...,
    ...,
    ...
]) # Sequential构建方式

inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
# Functional API构建方式
```

##### 训练

```python
model.compile(optimizer='adam',
         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
         metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)
```

##### 保存和加载

```python
model.save("mnist_model.h5") # h5格式 通过文件名后缀指定
model.save("mnist_model") # SavedModel格式 

h5_model = tf.keras.models.load_model("mnist_model.h5") # 加载h5模型
sm_model = tf.keras.models.load_model("mnist_model") # 加载SavedModel模型
```

### Fashion MNIST dataset

原始MNIST数据集过于简单，我们用普通的全连接层就能得到一个不错的训练结果，无法体现出CNN的优势

使用Fashion MNIST数据集和原始MNIST数据集对比，同一个模型的准确率都会下降10%左右，可见训练的难度提高了

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230117171327036.png" alt="image-20230117171327036" style="zoom:50%;" />

[用TensorFlow2训练Fashion MNIST数据集的分类问题](.\experts\code_sample\chapter-2)



## 方案设计篇

### 背景

AI新零售是消费方式的革新，逆向牵引生产变革，掌握数据、了解消费者需求后再生产是基本形态，通过数据和商业逻辑的深度结合，引领消费升级。

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230118142518940.png" alt="image-20230118142518940" style="zoom: 80%;" />

在零售行业，AI可以贯穿全流程提供更好的服务，如AR技术模拟穿衣、网购的实时翻译技术等等

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230118142636227.png" alt="image-20230118142636227" style="zoom:80%;" />

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230118142834989.png" alt="image-20230118142834989" style="zoom:80%;" />

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230118142907723.png" alt="image-20230118142907723" style="zoom:80%;" />

### 线下品牌之间的存量竞争

广告位、SKU稽查监管、竞品分析等方面目前都存在人工低效的问题，这些是品牌方的普遍需求

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230118143229214.png" alt="image-20230118143229214" style="zoom:67%;" />

对于零售品牌的管理者来说，销售指标完不成、商品易滞销、市场竞争惨烈、无法获取第一手市场信息、海量数据录入、同步不及时、难于做决策等等问题，导致的商机损失高达1000亿美元。

对于消费者来说，商品断货、商品关注不方便等问题也降低了消费的欲望。

### 解决方案

执行线下的门店拍照+稽核抽查+正负反馈对于品牌方来说效率过低，因为抽查比例低、覆盖门店少、人工费用高、全局把握难

利用AI+大数据技术实现品牌管理的智能看板，实现线下零售向数字化平台转型是解决这个问题的长期目标

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230118144019526.png" alt="image-20230118144019526" style="zoom:67%;" />

短期目标：自动化陈列审核和促销管理

### 业务落地

#### 第一步：货架数字化

货架、SKU和陈列都可以数字化成为一系列方便管理的item

![image-20230118144624242](C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230118144624242.png)

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230118144556837.png" alt="image-20230118144556837" style="zoom:80%;" />

#### 第二步：量化销售指标

##### 分销达标

品牌方对于SKU在货架陈列数量、摆放位置、摆放纯度等指标提出要求

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230118144927585.png" alt="image-20230118144927585" style="zoom:67%;" />

##### 新品上架陈列稽查

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230118204615292.png" alt="image-20230118204615292" style="zoom:50%;" />

##### 陈列激励

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230118204646494.png" alt="image-20230118204646494" style="zoom:50%;" />

#### 第三步：设计基于深度学习的AI流水线方案

货架商品检测，先检测出每一个item的位置信息Bbox，再根据训练的模型进行SKU分类

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230118204820816.png" alt="image-20230118204820816" style="zoom:50%;" />

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230118205005025.png" alt="image-20230118205005025" style="zoom:50%;" />

但此处存在很多难点，例如细分品类无法通过图形进行分辨（如不同口味），需要添加一层细分的分类器；例如规格信息（如净含量等）肉眼都无法识别，需要结合标签具体信息（物体实际高度、宽度等）作为特征训练。

#### 第四步：方案交付——支持在线识别和API调用的AI SaaS

showcase

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230118205503068.png" alt="image-20230118205503068" style="zoom:50%;" />

##### 通用物品识别平台架构

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230118205550944.png" alt="image-20230118205550944" style="zoom:50%;" />

### 商品识别AI+业务流水线

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230118205739376.png" alt="image-20230118205739376" style="zoom: 50%;" />
