## 环境依赖

基本依赖：python pip和virtualenv

基本命令：ls，cls

两个命令行：win+R、命令提示符+以管理员方式运行（有很多下载权限）

pip install报proxy error：要关掉VPN或者机场



## 虚拟环境virtualenv

安装virtualenv踩坑记录：https://blog.csdn.net/chenhanxuan1999/article/details/100675337?utm_medium=distribute.pc_relevant.none-task-blog-baidujs-9

windows下使用virtualenv的方法：https://blog.csdn.net/liuchunming033/article/details/46008301

---

创建虚拟环境：

```shell
>>> cd C:\Users\Lenvov\Documents\PythonVirtualEnv
>>> virtualenv venv
```

激活虚拟环境：

```shell
>>> venv\Scripts\activate
(venv) >>> deactivate
```

虚拟环境的jupyter notebook：

```shell
(venv) >>> pip install jupyter
(venv) >>> python -m ipykernel install --user --name=venv # 创建venv虚拟环境的kernal
(venv) >>> jupyter kernelspec list # 查看kernel数
```



## TensorFlow的整体架构

<img src=".\images\image-20230104163958852.png" alt="image-20230104163958852" style="zoom: 50%;" />

<img src=".\images\image-20230104164033823.png" alt="image-20230104164033823" style="zoom:50%;" />

<img src=".\images\image-20230104165300888.png" alt="image-20230104165300888" style="zoom:50%;" />

TF执行原理：任务队列+拓扑排序，将入度为0的任务放入队列中执行



## 张量（Tensor）

### 概念

拥有相同数据类型的多维数组

* 张量是用来表示多维数据的
* 执行操作时的输入或输出数据
* 用户通过执行操作来创建或计算张量
* 张量的形状不一定再编译时确定，可以在运行时通过形状推断计算得出

### 特殊张量

> tf.constant：不可变的张量
>
> tf.placeholder：多维数据的外壳
>
> tf.Variable：常驻内存，维护状态的张量

### 例子

```python
tf.Variable([[7],[11]], tf.int16) # 创建2阶变量的例子
tf.zeros([10, 299, 299, 3]) # 创建4阶张量
# 注意：张量和变量不同的地方是，张量在用完后会被垃圾回收
```



## 变量（Variable）

### 概念

常驻于内存，常用于维护状态的特殊张量

### 变量生命周期

<img src=".\images\image-20230105103005568.png" alt="image-20230105103005568" style="zoom:50%;" />

> tf.train.Saver和tf.Variable用于不同场景，前者可以用于save或restore变量为内存的值
>
> 创建一个变量后，可能会通过checkpoint文件保存的旧值进行初始化，也可能是一个新值
>
> 最后一步是更新变量的值，并保存回checkpoint文件中



## 操作（Operation）

### 概念

<img src=".\images\image-20230105110122958.png" alt="image-20230105110122958" style="zoom: 50%;" />

操作的输入和输出是张量或操作（函数式编程）

### 操作分类及常用

操作一般分为计算、控制、占位符操作

<img src=".\images\image-20230105110636755.png" alt="image-20230105110636755" style="zoom:33%;" />

注意：不推荐在数据流图中使用逻辑控制的操作，因为真正执行的过程无法预测

<img src=".\images\image-20230105111101877.png" alt="image-20230105111101877" style="zoom: 33%;" />

占位符指的是等待输入的张量，只有填充后才可以参与运算，否则会报错



## 会话（Session）

### 概念

<img src=".\images\image-20230105114715127.png" alt="image-20230105114715127" style="zoom:50%;" />

### 执行方法

不一定显式调用sess.run()，通过以下两种方法也可以执行会话，执行过程会隐式调用sess.run()方法

<img src=".\images\image-20230105114949287.png" alt="image-20230105114949287" style="zoom:50%;" />

### TensorFlow会话执行原理

<img src=".\images\image-20230105122834687.png" alt="image-20230105122834687" style="zoom:50%;" />

<img src=".\images\image-20230105123222063.png" alt="image-20230105123222063" style="zoom: 50%;" />



## 优化器（Optimizer）

### 前置知识：损失函数

<img src=".\images\image-20230105124624118.png" alt="image-20230105124624118" style="zoom: 50%;" />

### 前置知识：经验风险

<img src=".\images\image-20230105124847853.png" alt="image-20230105124847853" style="zoom:50%;" />

### 前置知识：结构风险

<img src=".\images\image-20230105132040517.png" alt="image-20230105132040517" style="zoom:50%;" />

<img src=".\images\image-20230105132216548.png" alt="image-20230105132216548" style="zoom: 50%;" />

### 前置知识：优化算法

<img src=".\images\image-20230105132536866.png" alt="image-20230105132536866" style="zoom:50%;" />

<img src=".\images\image-20230105132837125.png" alt="image-20230105132837125" style="zoom: 50%;" />

### TensorFlow训练机制

<img src=".\images\image-20230105133046923.png" alt="image-20230105133046923" style="zoom:50%;" /> 

### TensorFlow优化器

<img src=".\images\image-20230105133603282.png" alt="image-20230105133603282" style="zoom:50%;" />

直接调用minimize()等同于调用默认条件下的三部曲

上图中的SyncReplicasOptimizer是分布式场景下的优化器



## TensorFlow实战：房价预测

### 理论知识

#### 前置知识：监督学习

<img src=".\images\image-20230105171627362.png" alt="image-20230105171627362" style="zoom: 50%;" />

#### 前置知识：监督学习典型算法

<img src=".\images\image-20230105172131794.png" alt="image-20230105172131794" style="zoom: 33%;" />

#### 前置知识：线性回归

<img src=".\images\image-20230105172337077.png" alt="image-20230105172337077" style="zoom:33%;" />

<img src=".\images\image-20230105173111014.png" alt="image-20230105173111014" style="zoom:33%;" />

#### 前置知识：梯度下降

<img src=".\images\image-20230105173711263.png" alt="image-20230105173711263" style="zoom: 33%;" />

#### 前置知识：多变量线性回归及其梯度下降

<img src=".\images\image-20230105173818940.png" alt="image-20230105173818940" style="zoom: 33%;" />

<img src=".\images\image-20230105173942394.png" alt="image-20230105173942394" style="zoom: 33%;" />

#### 前置知识：归一化

<img src=".\images\image-20230105174602867.png" alt="image-20230105174602867" style="zoom: 33%;" />

```python
def normalize_feature(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())
```

<img src=".\images\image-20230105174741499.png" alt="image-20230105174741499" style="zoom: 33%;" />

#### 训练模型的工作流

<img src=".\images\image-20230106113249211.png" alt="image-20230106113249211" style="zoom: 50%;" />

##### step1 数据准备工作

由于我们前面提到的标量部分单独添加比较麻烦，我们用矩阵相乘来代替的时候需要一个恒为1的输入值x0

<img src=".\images\image-20230106120422032.png" alt="image-20230106120422032" style="zoom: 33%;" />

##### step2 构建数据流图

<img src=".\images\image-20230106122709168.png" alt="image-20230106122709168" style="zoom: 33%;" />

+ get_variable方法，可以生成为一个抽象节点，实现变量的复用
+ 学习率一般为超参数，此例子中设置为一个常量值
+ 最小二乘法采用矩阵自相乘（加一个转置操作）的方式实现，比较巧妙

##### step3 创建会话

<img src=".\images\image-20230106123544967.png" alt="image-20230106123544967" style="zoom: 33%;" />

- 本例中训练集小，采用全量数据训练，而不是批处理
- 这里with方法方便回收资源



#### TensorBoard看板

<img src=".\images\image-20230106124135771.png" alt="image-20230106124135771" style="zoom:50%;" />

<img src=".\images\image-20230106182807384.png" alt="image-20230106182807384" style="zoom: 33%;" />

主要过程就是FileWriter实例从Session中获取内容并写入Event file中，供TensorBoard使用

<img src=".\images\image-20230106183048313.png" alt="image-20230106183048313" style="zoom: 33%;" />

使用的序列化工具是ProtoBuf



#### TensorBoard的使用

##### step1 创建数据流图

为了让数据流图结构更清晰，我们会用name_scope来抽象每个步骤为一个节点

<img src=".\images\image-20230106210532326.png" alt="image-20230106210532326" style="zoom:33%;" />

##### step2 创建FileWriter实例

记得关闭输出流

<img src=".\images\image-20230106210517877.png" alt="image-20230106210517877" style="zoom: 33%;" />

##### step3 启动TensorBoard

<img src=".\images\image-20230106210501108.png" alt="image-20230106210501108" style="zoom: 33%;" />



### 实际操作

<img src=".\images\image-20230106211100591.png" alt="image-20230106211100591" style="zoom:33%;" />

[项目地址](.\geektime_TensorFlow-master\notebook-examples\chapter-4)



## TensorFlow实战：MNIST识别手写体数字

### 理论知识

#### MNIST数据集介绍

一套手写数字的图像数据集，60000个训练样例和10000个测试样例，每个图像为28*28的二阶数组，每个像素点存储的是该像素的灰度值，取值为0-255，为了加速训练，对数据做规范化后，每个像素位置存储的是0-1的灰度值数据

#### 加载MNIST数据集

```python
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data('mnist/mnist.npz')
```

#### 前置知识：Softmax网络

##### 感知机模型（Perceptron）

输出的y值如果超过上阈值，则输出1；如果低于下阈值，则输出0

<img src=".\images\image-20230108121247763.png" alt="image-20230108121247763" style="zoom: 67%;" />

##### 神经网络（NN）

本质上是一个多层感知机，因为现实生活中的分类问题往往不是二分这么简单，将多层神经元互相连接，上一层的输出作为下一层的输入，可以解决很多复杂的问题

<img src=".\images\image-20230108121533008.png" alt="image-20230108121533008" style="zoom:50%;" />

#### 前置知识：线性不可分与激活函数

<img src=".\images\image-20230108121802284.png" alt="image-20230108121802284" style="zoom:67%;" />

为了实现神经网络的非线性建模能力，解决一些线性不可分的问题，我们通常使用激活函数来引入非线性因素。激活函数都采用非线性函数，常用的有Sigmoid、tanh、ReLU等

<img src=".\images\image-20230108121836739.png" alt="image-20230108121836739" style="zoom: 67%;" />

#### 前置知识：全连接层（full connected layers）

全连接层是一种对输入数据直接做线性变换的线性计算层，用于学习输入和输出数据之间的变换关系。一般可以作为：

+ 特征提取层，实现特征融合
+ 最终分类层，每个神经元的输出值代表了每个输出类别的概率

<img src=".\images\image-20230108122156133.png" alt="image-20230108122156133" style="zoom:80%;" />

#### 前置知识：前向传播

特别注意权重符号的表示

<img src=".\images\image-20230108122344028.png" alt="image-20230108122344028" style="zoom: 50%;" />

<img src=".\images\image-20230108122502214.png" alt="image-20230108122502214" style="zoom:50%;" />

#### 前置知识：后向传播（Back Propagation）

利用链式法则，加速计算参数梯度值的算法，一般使用计算图的方式，进行模块化的运算

>  如何计算梯度？
>
> <img src=".\images\image-20230108122917096.png" alt="image-20230108122917096" style="zoom: 67%;" />

>  什么是计算图和模块化？
>
> 计算图是由单元运算和变量构成的计算流程图，每一个模块的计算都简单
>
> 图中黄色的量是后向传播过程中之前已经计算得到的值，绿色的量是前向传播过程中计算的值，红色框框住的是损失函数对于各训练参数的梯度值，基于此进行梯度下降的训练
>
> <img src=".\images\image-20230108123541534.png" alt="image-20230108123541534" style="zoom: 33%;" />

#### 前置知识：MNIST CNN网络

在MNIST Softmax网络下，网络是根据像素点的灰度值来学习特征，CNN则更接近人脑的学习方式，特征的学习是基于更高维度，学习到的准确率相对也更高（抗噪声能力强）。相比全连接层的Softmax，CNN网络有两个优势：

* 网络层更复杂，引入了卷积层、池化层、Dropout层、Flatten层等概念
* CNN数据规范化时不是直接将数据拉平成一维来学习，这在处理复杂（如像素点更多、彩色）的图片时需要训练的参数相对少得多

#### 前置知识：卷积层（Convolution Layer）

使用一系列卷积核与多通道输入数据做卷积的线性计算层，可以理解为一种“滤波”的过程，卷积核与输入数据作用之后提取出图像的特征，卷积层的提出是为了利用输入数据（如图像）中特征的局域性和位置无关性来降低整个模型的参数量

#### 前置知识：池化层（Pooling）

用于缩小数据规模的一种非线性计算层，可以降低特征维度，一般用于一个或多个卷积层后。有三个参数：

+ 池化类型，一般有最大池化和平均池化两种
+ 池化核的大小k
+ 池化核的滑动间隔s

如下图，池化核的大小为2*2，滑动间隔为2

<img src=".\images\image-20230109152143987.png" alt="image-20230109152143987" style="zoom: 80%;" />

#### 前置知识：Dropout层

一种常用的正则化方法，用于解决全连接层的参数量太大，容易发生过拟合的问题，每次迭代训练时，将神经元以一定的概率值暂时丢弃，不参与当前迭代的训练

<img src=".\images\image-20230109152411632.png" alt="image-20230109152411632" style="zoom:80%;" />

#### 前置知识：Flatten层

用于卷积和池化层后，将提取的特征摊平后输入全连接网络，与Softmax网络的输入层类似

<img src=".\images\image-20230109152529996.png" alt="image-20230109152529996" style="zoom:80%;" />

![image-20230109152555019](.\images\image-20230109152555019.png)

### 实际操作

[项目地址](.\geektime_TensorFlow-master\notebook-examples\chapter-5)



## TensorFlow实战：验证码识别

### 理论部分

#### 依赖库介绍

- Pillow：PIL的fork版本，是一个python图像库，支持广泛的文件格式，可以作为图像处理工具
- captcha：生成图像和音频验证码的开源工具库
- pydot：主要依赖pyparsing和GraphViz这两个工具库，pyparsing用于加载DOT文件（graph description language），GraphViz用于将图像渲染为PDF，PNG，SVG等格式文件
- flask：Python Web应用程序框架

```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello, World!"
```

```shell
$ env Flask_APP=hello.py flask run
```

#### 验证码

> 验证码是一种区分用户是计算机或人的程序，是一种反向图灵测试（计算机来考人类）

#### 

#### 处理输入数据

生成验证码数据集（RGB图片）-> 灰度图（公式计算） -> 规范化数据（/=255）

<img src=".\images\image-20230110161306124.png" alt="image-20230110161306124" style="zoom:67%;" />

#### 处理输出数据

One-hot编码：验证码转向量

<img src=".\images\image-20230110161404399.png" alt="image-20230110161404399" style="zoom:67%;" />

解码：模型输出向量转验证码（argmax）

<img src=".\images\image-20230110161441186.png" alt="image-20230110161441186" style="zoom:67%;" />

#### 模型结构设计

图像分类模型如AlexNet、VGG-16，本质上就是利用卷积将特征泛化，不断提高卷积层数，学习到的特征就更加具体，但卷积是一个黑盒，我们无法弄清每一个layer具体学得的是什么特征。

验证码识别的模型结构（可以用graphviz生成）：

<img src=".\images\image-20230110161853378.png" alt="image-20230110161853378" style="zoom:67%;" />

#### 前置知识：交叉熵（Cross-Entropy, CE）

我们用交叉熵作为模型的损失函数，在分类问题中的标配：softmax输出层+交叉熵损失函数。常用的CE有两种：Binary CE和Categorical CE，具体定义如下：

<img src=".\images\image-20230110162056097.png" alt="image-20230110162056097" style="zoom: 80%;" />

其中ti表示真实label（one-hot编码下的0或1），si表示预测值（发生概率，0-1）

对于二分类问题，CE定义为：

<img src=".\images\image-20230110165252688.png" alt="image-20230110165252688" style="zoom:80%;" />

#### Categorical CE Loss(Softmax Loss)

<img src=".\images\image-20230110165427785.png" alt="image-20230110165427785" style="zoom:80%;" />

#### Binary CE Loss(Sigmoid CE Loss)

<img src=".\images\image-20230110165601168.png" alt="image-20230110165601168" style="zoom:80%;" />

#### 前置知识：学习率（Learning rate）

![image-20230110165707007](.\images\image-20230110165707007.png)

#### 前置知识：优化器及其对比

##### SGD(Stochastic Gradient Descent)

最基础的梯度下降，每次下降为学习率*梯度值，有经验的工程师可以通过观察Loss等参数，适时调整学习率来获得更好的效果

<img src=".\images\image-20230110231431733.png" alt="image-20230110231431733" style="zoom:80%;" />

##### SGD-M(Momentum)

普通SGD遇到沟壑时容易陷入震荡，导致收敛速度变慢，引入动量可以加速SGD在正确方向的下降并抑制震荡，每次更新都与上一次的梯度有关

<img src=".\images\image-20230110232020952.png" alt="image-20230110232020952" style="zoom:67%;" />

##### Adagrad

相对SGD-M相比，Adagrad引入二阶动量。在训练参数的过程中，有的参数更新很频繁，有的不频繁，针对不频繁的，我们的学习率提高，针对频繁的，每次更新的步长小一些，使参数更新更稳定。

如下图，vt为二阶动量，关注的是从第一次到当前次的所有梯度值，是单调递增的，每次更新的时候下降的值为学习率/vt

<img src=".\images\image-20230110232548349.png" alt="image-20230110232548349" style="zoom:67%;" />

##### RMSprop

改进了Adagrad的问题，由于vt使单调递增的，会导致学习率越来越小，训练过程提前结束。RMSprop的改进就在于只关注某一段时间窗口内的梯度记录

<img src=".\images\image-20230110232556263.png" alt="image-20230110232556263" style="zoom:67%;" />

##### Adam（Adadelta）

结合前两者的优点，对一阶二阶都使用指数移动平均

<img src=".\images\image-20230110232604828.png" alt="image-20230110232604828" style="zoom:67%;" />

##### 对比

在鞍点的问题下，Adadelta>Adagrad>Rmsprop>>NAG>Momentum，SGD无法逃离鞍点获得更好的优化效果

<img src=".\images\image-20230110233816845.png" alt="image-20230110233816845" style="zoom:50%;" />

##### 验证码模型识别对比

###### 优化器

<img src=".\images\image-20230111153923152.png" alt="image-20230111153923152" style="zoom: 50%;" />

<img src=".\images\image-20230111154034215.png" alt="image-20230111154034215" style="zoom: 50%;" />

我们可以看到在训练集下Adam的效果不是最好，但这并不代表Adam优化器比不过其他优化器，相反，这些优化器在测试集下的表现对比，Adam是最好的。这说明Adam在一定程度上避免了过拟合，在参数调整上也做到了最优，见下图：

<img src=".\images\image-20230111154112578.png" alt="image-20230111154112578" style="zoom:50%;" />

<img src=".\images\image-20230111154142484.png" alt="image-20230111154142484" style="zoom:50%;" />

###### Loss函数

在训练集上表现：

<img src=".\images\image-20230111154624396.png" alt="image-20230111154624396" style="zoom:50%;" />

<img src=".\images\image-20230111154653693.png" alt="image-20230111154653693" style="zoom:50%;" />

在测试集上表现：

<img src=".\images\image-20230111154717440.png" alt="image-20230111154717440" style="zoom:50%;" />

<img src=".\images\image-20230111154817255.png" alt="image-20230111154817255" style="zoom:50%;" />

### 实际操作

[项目地址](.\geektime_TensorFlow-master\notebook-examples\chapter-6)

#### 模型部署

选用Flask轻量级框架

<img src=".\images\image-20230110234350107.png" alt="image-20230110234350107" style="zoom: 33%;" />

创建Flask实例，并编写访问接口，最后命名为app.py，在cmd中通过命令行部署服务（要在对应目录下）

```shell
$ export FLASK_ENV=development && flask run --host=0.0.0.0
```

然后可以通过浏览器直接访问（GET方法）或Curl命令行访问

<img src=".\images\image-20230110234425156.png" alt="image-20230110234425156" style="zoom: 67%;" />



##  TensorFlow实战：人脸识别

### 理论部分

#### 人脸识别背景知识

利用分析比较人脸特征信息进行身份鉴别的计算机技术，难点主要在于人脸结构相似（利于定位、不利于区分），外形不稳定（光照条件、遮盖物、年龄、姿态角度等因素），人脸识别（区分）之前首先是人脸检测（定位）。

#### 常见数据集

##### 人脸检测数据集

<img src=".\images\image-20230115155539734.png" alt="image-20230115155539734" style="zoom:67%;" />

##### 人脸识别数据集

<img src=".\images\image-20230115155422454.png" alt="image-20230115155422454" style="zoom:67%;" />

#### 人脸检测算法

目标是找出图像中所有的人脸对应的位置，主要是用大量的人脸和非人脸样本图像进行训练，得到一个二分类分类器，解决是否为人脸的问题。

人脸检测算法有3个发展阶段：

- 基于模板匹配的算法
- 基于AdaBoost的框架
- 基于深度学习的算法

##### 基于模板匹配的算法

最早期的技术，将人脸模板和被检测图像的各个位置进行匹配，判断这个位置上是否有人脸，类似的思路是滑动窗口

解决多角度人脸检测问题时，整个系统由两个神经网络构成，第一个用于估计人脸的角度，第二个用于判断是否为人脸

##### 基于AdaBoost框架的人脸检测算法

Boost算法是基于PAC（Probably Approximately Correct）学习理论而建立的一套集成学习（Ensemble Learning）算法。核心思想是利用多个简单的弱分类器，构建出高准确度的强分类器。

这个阶段的代表性框架就是VJ框架，使用简单的Haar-like特征和级联的AdaBoost分类器构造检测器，检测速度有两个数量级的提高，保持了很好的精度。但存在以下问题：

+ Haar-like特征是一种相对简单的特征，其稳定性较低
+ 弱分类器采用简单的决策树，容易过拟合。解决非正面的人脸情况效果一般
+ 基于VJ-cascade的分类器设计，进入下一个stage后，之前的信息都丢失了，分类器评价一个样本不会基于它在之前stage的表现，鲁棒性差

##### 基于深度学习的人脸检测算法

CNN在人脸检测问题的精度大幅超越AdaBoost框架，此前滑动窗口+卷积对窗口图像进行分类的计算量巨大，无法做到实时检测。当前已有的快速高效的深度学习算法有

- Cascade CNN
- MTCNN

###### Cascade CNN

Cascade CNN是传统技术和深度学习相结合的代表，和VJ人脸检测器一样，它包含了多个分类器，这些分类器也是采用级联结构进行组织。不同地方在于，Cascade CNN采用卷积网络作为每一级的分类器。

<img src=".\images\image-20230115162040543.png" alt="image-20230115162040543" style="zoom:80%;" />

###### MTCNN

<img src=".\images\image-20230115162204514.png" alt="image-20230115162204514" style="zoom:67%;" />

#### 人脸识别算法

1. 人脸检测（Face Detection）
2. 人脸对齐（Face Alignment）
3. 人脸特征表示（Feature Representation）

<img src=".\images\image-20230115163539531.png" alt="image-20230115163539531" style="zoom:50%;" />

人脸识别算法分为3个发展阶段：

* 早期算法：基于几何特征、模板匹配、子空间等
* 人工特征+分类器
* 基于深度学习的算法

##### 早期算法

将人脸图像当作一个高维向量，将其投影到低维空间后，通过低维向量区分人脸。常见的降维算法分为线性降维和非线性降维，线性降维技术有PCA和LDA，然而人脸在高维空间中的分布是非线性的，用非线性降维算法，流形学习和核方法。

<img src=".\images\image-20230115164516118.png" alt="image-20230115164516118" style="zoom:50%;" />

流形学习是一种非线性降维方法，它假设向量点在高维空间中的分布具有某些几何形状，在保持这些几何形状约束的前提下将向量投影到低维空间中，尽可能降低对数据分布的影响，如下图，瑞士卷降维成二维时保留了一定的分布

![image-20230115164758226](.\images\image-20230115164758226.png)

降维方法的缺点在于太过于依赖训练集和测试集的场景，泛化能力比较差。

##### 人工特征+分类器

人工特征是一项基于经验的工作，需要设计出有效区分不同人的特征，CV领域很多描述图像的特征如HOG、SIFT、Gabor、LBP（部分解决光照敏感问题）等都是这个阶段的产物。

分类器使用联合贝叶斯，它是对贝叶斯人脸的改进方法，选用LBP和LE作为基础特征，将人脸图像和差异表示为人因姿态、表情等导致的差异和不同人间的差异两个因素，用潜在变量组成的协方差建立两张人脸的关联。

MSRA的Feature Master主要以LBP为例，论述了高纬度特征和验证性能成正相关，人脸维度越高，验证的准确度就越高。

![image-20230115165915251](.\images\image-20230115165915251.png)

##### 基于深度学习算法

意识到CNN学习的卷积核效果优于“人工特征+分类器”的方案后，大家开始用CNN+海量人脸图片来提取特征，前期主要精力集中在神经网络结构和输入数据设计方面，后期主要研究人脸损失函数的改进。

两个在这个阶段具有代表性的模型是Facebook的DeepFace和Google的FaceNet

###### DeepFace

<img src=".\images\image-20230115170650709.png" alt="image-20230115170650709" style="zoom: 50%;" />

###### FaceNet

<img src=".\images\image-20230115170725080.png" alt="image-20230115170725080" style="zoom:50%;" />

### 实际操作

#### 解析FaceNet人脸识别模型

<img src=".\images\image-20230115171156671.png" alt="image-20230115171156671" style="zoom:50%;" />

##### Deep Architecture

  <img src=".\images\image-20230115213512726.png" alt="image-20230115213512726" style="zoom:67%;" />

架构、图像质量、特征空间维度、训练数据量等对准确度都有影响

##### Triplet Loss

<img src=".\images\image-20230115171459079.png" alt="image-20230115171459079" style="zoom:80%;" />

#### FaceNet的创新与突破

1. 使用Triplet Loss替代Softmax Loss
2. 模型输出128维人脸特征向量而不是人脸分类结果
3. 准确率在LFW上达到99.63%
4. 模型“瘦身”，减少了参数和计算量
5. 通用性好，可应用到多种场景

[项目地址](.\geektime_TensorFlow-master\notebook-examples\chapter-7)



## 番外篇

### ML全生态

<img src=".\images\image-20230116112540791.png" alt="image-20230116112540791" style="zoom:67%;" />

### TFX-基于TensorFlow的端到端机器学习平台

<img src=".\images\image-20230116112714079.png" alt="image-20230116112714079" style="zoom:80%;" />

### Kubeflow-Kubernetes与Dataflow的结合

<img src=".\images\image-20230116112852885.png" alt="image-20230116112852885" style="zoom:80%;" />

为了方便MLE（Machine Learning Engineer主要任务是中间三个模块）完成全流程工作，即模型训练、模型可视化、模型服务，Kubeflow利用k8s的容器特性完成这些步骤，部署时也利用Nginx做负载均衡。

<img src=".\images\image-20230116113243376.png" alt="image-20230116112852885" style="zoom:80%;" />



在训练过程，可以设置PS和Worker的数量实现并行化训练，生成的tf.events和saved_model.pb文件分别用于模型可视化和模型服务。



## Reference

内容来自于[极客时间-TensorFlow 快速入门与实战](https://time.geekbang.org/course/intro/153)

bilibili视频[BV1yG411x7Cc](https://www.bilibili.com/video/BV1yG411x7Cc/)、[BV1mZ4y1R76t](https://www.bilibili.com/video/BV1mZ4y1R76t/)

特别感谢~
