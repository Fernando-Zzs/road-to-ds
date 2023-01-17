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

