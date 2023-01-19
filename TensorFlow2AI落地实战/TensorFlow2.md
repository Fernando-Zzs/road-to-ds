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



## 商品检测篇

### 基础

#### 目标检测及相关概念

在一张图片中框出经过训练的目标并打上对应的label的过程叫做目标检测

##### Ground Truth/Bounding Box

![image-20230119144450032](C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230119144450032.png)

Ground Truth：人对数据集标注的框

Bounding Box：模型预测的框

##### 评估标准：IoU（Intersection over Union）

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230119144727929.png" alt="image-20230119144727929" style="zoom: 67%;" />

##### 评估标准：准确率和召回率

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230119144928479.png" alt="image-20230119144928479" style="zoom:80%;" />

- TP：预测框识别正确的数量
- FP：预测框没有识别到的正确物体的数量
- FN：预测框识别错误的数量

TP+FP是Ground Truth，TP+FN是图中预测框的个数

识别正确和错误的判断是基于IoU设置的阈值和分类物体时置信度阈值

##### 评估标准：mAP（mean Average Precision）

实际上准确率和召回率是存在一定矛盾的，准确率导向的预测相对谨慎，漏判的情况就会多；召回率导向的预测相对宽松，容易错判。为了用一个指标衡量两者的好坏，就有了AP的概念。

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230119145956150.png" alt="image-20230119145956150" style="zoom:67%;" />

假设检测的可能物体有多个类别，则mAP是所有类别的AP的平均值

#### 目标检测发展

![image-20230119150138788](C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230119150138788.png)

2012年是一个分水岭，深度学习取代了经验主义的特征工程（根据图片特性设置滤波器）；有两种发展方向，分别是两阶段检测和一阶段检测。落地到工业界的是比较先进的YOLOv3和Retina-Net，各自有合适的适用场景。

Retina-Net和YOLO的对比

YOLO首先提出一阶段，YOLOv2增加一些tricks，YOLOv3引入了残差模块、FPN等改进，准确率越来越高、时间越来越短

Retina-Net解决了样本不平衡、类不平衡的问题，多尺度FeatureMap的学习、难度大的负样本学习，提出新的Loss函数Focal Loss



#### R-CNN系列二阶段模型：一个回归问题+一个分类问题

##### Slow R-CNN

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230119151447263.png" alt="image-20230119151447263" style="zoom:50%;" />

过程：

1. 输入一张图片
2. 选出候选框。Proposal method是指Selective Search，是根据颜色、纹理等特征筛选出大约2k个候选框用于后期卷积（已经比暴力枚举少得多）
3. 弯曲成特定大小放入卷积层抽取特征
4. FeatureMap存入硬盘中
5. 读取硬盘再输出到支持向量机中做分类

问题：候选框大量冗余、卷积时采用VGG-Net-16这一重型网络、CNN特征提取后存入硬盘，再从硬盘中读取、用SVM做分类。时间、计算资源开销太大。

表现：在当时最好的GPU上也要47秒检测一张图片



##### Fast R-CNN

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230119152044213.png" alt="image-20230119152044213" style="zoom: 50%;" />

比较：相比前一代增加了Bbox框的回归任务，引入了RoI Pooling layer、改进了存盘部分

过程：所有候选框输入到全卷积的网络得到FeatureMap，这些FeatureMap经过Pooling layer会统一成一个同尺寸的RoI，统一尺寸的RoI输入到两个全连接层，最终输出用于分类的softmax和用于Bbox回归的Loss，是一种多任务的学习模式

表现：可以达到几秒钟检测一张图片

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230119152340200.png" alt="image-20230119152340200" style="zoom: 67%;" />



##### Faster R-CNN

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230119152801592.png" alt="image-20230119152801592" style="zoom:50%;" />

对比：引入Region Proposal Network取代Selective Search、同时引入Anchor作为不同尺寸图片的适配（在后来的YOLO也有用到）



#### YOLO系列一阶段模型：回归分类一起完成

##### YOLOv1：首个深度学习的一阶段检测器

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230119153244939.png" alt="image-20230119153244939" style="zoom: 50%;" />

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230119153337391.png" alt="image-20230119153337391" style="zoom:50%;" />

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230119153400207.png" alt="image-20230119153400207" style="zoom:50%;" />

参照了GoogleNet的网络结构

过程：

1. 将图像resize成448*448
2. 放到卷积网络中抽取特征向量，将图像分割为5\*5或7\*7（S）的格子图，每一个格子都输出识别出的目标物体（B）的四个坐标+置信度的参数，以及可能属于的类别（C） 
3. 经过卷积后接入全连接层，最终输出的向量为S\*S\*（B\*5+C）

优点：作为首个深度学习的一阶段检测器，在准确率和误检率都很高的情况下，检测速度是Slow R-CNN的上千倍

缺点：

* 检测目标密集场景下不适合，召回率比较低，存在漏检的情况（划分了格子的原因）
* 同一目标出现新的长宽比时泛化能力弱（有Risize的原因）
* 位置的精准性比较差（还是格子的原因）

##### YOLOv2：更快、更好、更强

各方面的提升

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230119160615602.png" alt="image-20230119160615602" style="zoom: 50%;" />

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230119160704566.png" alt="image-20230119160704566" style="zoom: 50%;" />

优化后的网络结构

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230119160811579.png" alt="image-20230119160811579" style="zoom:50%;" />



#### RCNN与YOLO系列的比较

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230119160911372.png" alt="image-20230119160911372" style="zoom:50%;" />

可以见到YOLO的速度虽然快，但准确率不如RCNN系列，因为没有了region proposal的检测算法，为了兼具快、准两个优点，我们需要找到准确率不如two-stage的根本原因：样本类别不均衡（正负样本不平衡），也就是一张图的candidate location太多，负样本占总的loss的大部分，而且多是容易分类的。

Retina-Net中的Focal Loss这个函数可以通过减少易分类样本的权重，使得模型在训练时更专注于难分类的样本。

#### Focal Loss

Focal Loss本质上是交叉熵损失函数CE的变体，原始的交叉熵对于每一个样本的权重是一样的

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230119161243981.png" alt="image-20230119161243981" style="zoom:50%;" />

首先引入超参数αt，可以通过设定αt的值来控制正负样本对总的loss的共享权重。αt取比较小的值来降低负样本（多的那类样本）的权重，反之可以降低正样本的权重

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230119163624974.png" alt="image-20230119163624974" style="zoom: 67%;" />

但还无法解决分类难易程度的权重，于是有了Focal Loss的完全体：

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230119163928484.png" alt="image-20230119163928484" style="zoom:80%;" />

其中超参数γ也称作focusing parameter，加上这个调制因子目的是通过减少易分类样本的权重，从而使得模型在训练时更专注于难分类的样本

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230119161234078.png" alt="image-20230119161234078" style="zoom: 50%;" />

横坐标是pt，即正样本的概率

**这里介绍下focal loss的两个重要性质：**

1、当一个样本被分错的时候，pt是很小的，那么调制因子（1-Pt）接近1，损失不被影响；当Pt→1，因子（1-Pt）接近0，那么分的比较好的（well-classified）样本的权值就被调低了。因此调制系数就趋于1，也就是说相比原来的loss是没有什么大的改变的。当pt趋于1的时候（此时分类正确而且是易分类样本），调制系数趋于0，也就是对于总的loss的贡献很小。

**2、**当γ=0的时候，focal loss就是传统的交叉熵损失，**当γ增加的时候，调制系数的影响也会增加。** 专注参数γ平滑地调节了易分样本调低权值的比例。γ增大能增强调制因子的影响，实验发现γ取2最好。直觉上来说，**调制因子减少了易分样本的损失贡献，拓宽了样例接收到低损失的范围。**当γ一定的时候，比如等于2，一样easy example(pt=0.9)的loss要比标准的交叉熵loss小100+倍，当pt=0.968时，要小1000+倍，但是对于hard example(pt < 0.5)，loss最多小了4倍。这样的话hard example的权重相对就提升了很多。这样就增加了那些误分类的重要性

**focal loss的两个性质算是核心，其实就是用一个合适的函数去度量难分类和易分类样本对总的损失的贡献。**

参考：https://zhuanlan.zhihu.com/p/49981234

为了同时达到易于调整正负样本的权重、难易分类样本的权重，采用如下公式：

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230119165624292.png" alt="image-20230119165624292" style="zoom:67%;" />

#### RetinaNet的介绍与成果

结构上采用的是Resnet+FPN（feature pyramid net）作为backbone，多规格的卷积特征金字塔每一层都有classification和regression of boxes两路subnet

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230119170449254.png" alt="image-20230119170449254" style="zoom: 50%;" />

RetinaNet训练成果：

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230119165757320.png" alt="image-20230119165757320" style="zoom: 50%;" />

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230119170918263.png" alt="image-20230119170918263" style="zoom:50%;" />

#### YOLOv3：小目标识别大提升

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230119171706023.png" alt="image-20230119171706023" style="zoom:33%;" />

关于选择YOLOv3还是RetinaNet需要考虑的因素：

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230119171804206.png" alt="image-20230119171804206" style="zoom:50%;" />

### 应用

#### 监测数据标注

labelImg的常用快捷键

<img src="C:\Users\Lenvov\AppData\Roaming\Typora\typora-user-images\image-20230119201020894.png" alt="image-20230119201020894" style="zoom: 67%;" />
