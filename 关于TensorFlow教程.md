---
typora-root-url: D:\typora图片总
---






# TensorFlow教程

## 一、基本概念：

 **Tensorflow的设计理念称之为计算流图，在编写程序时，首先构筑整个系统的graph，代码并不会直接生效，这一点和python的其他数值计算库（如Numpy等）不同，graph为静态的，类似于docker中的镜像。然后，在实际的运行时，启动一个session，程序才会真正的运行。这样做的好处就是：避免反复地切换底层程序实际运行的上下文，tensorflow帮你优化整个系统的代码。我们知道，很多python程序的底层为C语言或者其他语言，执行一行脚本，就要切换一次，是有成本的，tensorflow通过计算流图的方式，帮你优化整个session需要执行的代码，还是很有优势的。**



-  使用图（graphs）来表示计算任务，在被称之为会话（Session）的上下文（context）中执行图；

-   图中的节点称之为op（operation），一个op获得0个或者多个Tensor;

-   使用tensor表示数据,一个tensor可以看作是一个n维的数组或列表；

-    通过变量（Variable）维护状态；

-    使用feed和fetch可以为任意的操作赋值或者从中获取数据

TensorFlow三个基础核心概念：计算图、Tensor、Session

   ![](/tensorflow基本概念1.jpg)        

**1、计算图：**

在TensorFlow中，计算图是一个有向图，用来描述计算节点以及计算节点之间的关系，所以在TensorFlow中我们存储一个值或者数组的时候，存的其实是这个值或者数组的计算图而不是其本身的数字。我们可以用写一个简单的例子来验证一下：

**关于计算图的操作**

1、新建计算图:g=tf.Graph()，但是不同计算图上的张量是不能共享的,这个是存在于变量
2、指定计算图的使用的device:with g.device("/gpu:0"):
3、设置默认计算图:with g.as_default:
4、在会话中可以指定使用的计算图:with tf.Session(graph=g1):
对于以上操作用代码说话

```python
import tensorflow as tf
g1=tf.Graph()
with g1.as_default():
	a=tf.constant([1,2,3],name="a")#用常量试试看
    b=tf.get_variable('b',initializer=tf.constant_initializer()(shape = [1]))
    #用变量试试看
g2=tf.Graph()
with g2.as_default():
    a=tf.constant([2,3],name="a")#用常量试试看
    b=tf.get_variable('b',initializer=tf.constant_initializer()(shape = [3]))
    #用常量试试看
with tf.Session(graph=g2) as sess:
    with g1.device("/cpu:0"):
        tf.global_variables_initializer().run()
        c=a+1
        print("常量的情况下",sess.run(c))
with tf.variable_scope("", reuse=True):
            print("变量情况下",sess.run(tf.get_variable("b"))) 
       
with tf.Session(graph=g2) as sess:
    with g2.device("/gpu:0"):
       tf.global_variables_initializer().run()
        c=a+1
        print("常量的情况下",sess.run(c))
        with tf.variable_scope("", reuse=True):
        print("变量情况下",sess.run(tf.get_variable("b")))​           
```

**2、张量：**

张量（tensor）可以简单理解为多维数组。其中零阶张量表示标量（scalar），也就是一个数；一阶张量为向量（vector），也就是一维数组；第n阶张量可以理解为一个n维数组。但是张量在Tensorflow中的实现并不是直接采用数组的形式，它只是对Tensorflow中运算结果的引用。在张量中并没有真正保存数字，它保存的是如何得到这些数字的计算过程。

**3、会话：**

在TensorFlow中，计算图的计算过程都是在会话下进行的，同一个会话内的数据是可以共享的，会话结束计算的中间量就会消失。



------

## 二、tensorflow中的数据类型

| 数据类型     | Python 类型  | 描述                                              |
| ------------ | ------------ | :------------------------------------------------ |
| DT_FLOAT     | tf.float32   | 32 位浮点数.                                      |
| DT_DOUBLE    | tf.float64   | 64 位浮点数                                       |
| DT_INT64     | tf.int64     | 64 位有符号整型.                                  |
| DT_INT32     | tf.int32     | 32 位有符号整型.                                  |
| DT_INT16     | tf.int16     | 16 位有符号整型.                                  |
| DT_INT8      | tf.int8      | 8 位有符号整型.                                   |
| DT_UINT8     | tf.uint8     | 8 位无符号整型.                                   |
| DT_STRING    | tf.string    | 可变长度的字节数组.每一个张量元素都是一个字节数组 |
| DT_BOOL      | tf.bool      | 布尔型.                                           |
| DT_COMPLEX64 | tf.complex64 | 由两个32位浮点数组成的复数:实数和虚数.            |
| DT_QINT32    | tf.qint32    | 用于量化Ops的32位有符号整型.                      |
| DT_QINT8     | tf.qint8     | 用于量化Ops的8位有符号整型.                       |
| DT_QUINT8    | tf.quint8    | 用于量化Ops的8位无符号整型.                       |



------

## 三、实例演示

### 1、创建图、启动图

```python
import tensorflow as tf
#创建两个常量op
m1=tf.constant([[3,3]])
m2=tf.constant([[2],[3]])
#创建一个矩阵乘法op，把m1和m2传入
product=tf.matmul(m1,m2)
#定义一个会话，启动默认图
sess=tf.Session()
#调用sess的run方法来执行矩阵乘法op
#run(product)触发了图中的3个op
result=sess.run(product)
print(result)
sess.close
```

**对上面图操作变形：**

```python
import tensorflow as tf
#创建两个常量op
m1=tf.constant([[3,3]])
m2=tf.constant([[2],[3]])
#创建一个矩阵乘法op，把m1和m2传入
product=tf.matmul(m1,m2)
#定义一个会话，启动默认图
with tf.Session() as sesss:
	#调用sess的run方法来执行矩阵乘法op
	#run(product)触发了图中的3个op
	result=sess.run(product)
	print(result)
```

### 2、变量操作

```python
#创建一个变量并初始化为0
state = tf.Variable(0,name='counter')
#创建一个op，作用是使state+1
new_value = tf.add(state,1)
#创建一个赋值op
update = tf.assign(state,new_value)

#变量初始化
init = tf.global_variable_initializer()
with tf.Session() as sess:
    sess.run(init)
    print (sess.run(state))
    for _ in range(5):
        sess.run(updata)
        print (sess.run(state))
```

### 3、Fetch and Feed

```python
import tensorflow as tf
#Fetch,可以在会话中同时运行多个op
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

add = tf.add(input2,input3)
mul = tf.multiply(input1,add)

with tf.Session() as sess:
    result = sess.run([mul,add])
    print(result)
```

```python
import tensorflow as tf
#Feed
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)

with tf.Session() as sess:
    #feed的数据以字典形式传入
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))
```

### 4、tensorflow简单示例

```Python
import tensorflow as tf
import numpy as np

#使用numpy生成100个随机点
x_data = np.random.rand(100)
y_data = x_data*0.1 + 0.2

#构造一个线性模型
b = tf.Variale(0.)
k = tf.Variable(0.)
y = k*x_data +b 

#二次代价函数
loss = tf.reduce_mean(tf.square(y_data - y))
#定义一个梯度下降法来进行训练的优化器
optimizer = tf.train.GradientDescentOptimizer(0.2)
#最小化代价函数
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step%20 == 0:
            print(step,sess.run([k,b]))
```

由此可以得出，tensorflow代码部分有：

**输入训练的数据，确定模型，确定代价函数，确定优化方式，建立会话，初始化变量，启动会话**

### 5、非线性回归

```Python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#使用numpy生成200个随机点，通过np.newaxis在列上增加维度，最后生成的则为200行1列的数组
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
#生成正态分布的噪声，其形状和x_data一样。
noise = np.random_normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noise

#定义两个placeholder，在模型中占位，并未把数据传入模型，只是分配必要的内存。其形状和上面的数据相同
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

#定义神经网络中间层
#这里的[1,10]，说明输入层是一个数，中间层是10个单元
Weights_L1 = tf.Variable(tf.random_normal([1,10]))
biases_L1 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1 = tf.matmul(x,Weights_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)

#定义神经网络输出层
#这里的[10,1]，说明中间层有十个单元，输出层只有一个单元
Weights_L2 = tf.Variable(tf.random_normal([10,1]))
biases_L2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 = tf.matmul(L1,Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

#二次代价函数，预测值与真实值的平方再求平均
loss = tf.reduce_mean(tf.square(y-prediction))
#使用梯度下降法训练，学习率是0.1，优化的过程中会改变w,b权重的值
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    #变量初始化
    sess.run(tf.global_variables_initializer())
    #训练2000次
    for _ in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
        
    #获得预测值
    prediction_value = sess.run(prediction,feed_dict={x:x_data})
    #画图
    plt.figure()
    #生成散点图
    plt.scatter(x_data,y_data)
    #r-代表红色，lw是线宽
    plt.plot(x_data,prediction_value,'r-',lw=5)
    plt.show()
```

### 6、MNIST数据集分类简单版本

**(注：此版本是简化版本，只有输入层和输出层，没有隐藏层)**

MNIST数据集官网：http://yann.lecun.com/exdb/mnist/

下载下来的数据集被分成两部分：60000行的训练数据集（train-images-idx3-ubyte.gz和train-labels-idx1-ubyte.gz）和10000行的测试数据集（t10k-images-idx3-ubyte.gz和t10k-labels-idx1-ubyte.gz）



每一张图片包含28\*28个像素，把这个数组展开成一个向量，长度是28*28=784。因此在MNIST训练数据集中，mnist.train.images是一个形状为[60000,784]的张量，第一个维度数字用来索引图片，第二个维度数字用来索引每张图片中的像素点。图片里的某个像素的强度值介于0-1之间。

![](/MNIST1.png)

![](/MNIST2.png)

MNIST数据集的标签是介于0-9的数字，我们要把标签转化为“one-hot vectors".

一个one-hot向量除了某一位数字是1以外，其余维度数字都是0，比如标签0将表示为（[1,0,0,0,0,0,0,0,0,0]）。

因此，mnist.train.labels是一个[60000,10]的数字矩阵。

![](/MNIST3.png)

**Softmax函数：**

MNIST的结果是0-9，模型可能推测出一张图片是数字9的概率是80%，是数字8的概率是10%，然后其他数字的概率更小，总体概率加起来等于1。这是一个使用softmax回归模型的经典案例。

softmax模型可以用来给不同的对象分配概率。

![](/softmax.jpg)

比如，输出的结果为：[1,5,3]

$e^1=2.718​$                  

$e^5=148.413$

$e^3=20.086$

$e^1+e^5+e^3=171.217$

![](/softmax2.jpg)

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
#每个批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义两个placeholder,这里的None取决于批次的大小
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#创建一个简单的神经网络
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

#二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))

#使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

'''
argmax返回一维张量中最大的值所在的位置,通过equal函数判断，预测结果和标签中的最大值，
是否是在相同的位置，即是否判断的数字是否正确。
结果存放在一个布尔型的列表中，存放的是Ture和False
'''
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
#求准确率
#tf.cast将Ture转化为1.0，将False转化为0
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    #迭代21个周期，每个周期，根据批次的大小，训练不同的次数n_batch,即每个周期训练一次所有图片
    for epoch in range(21):
        for batch in range(n_batch):
            #mnist.train.next_batch每次从训练数据集里导入100张图
            #图片存在batch_xs,标签放在batch_ys
            batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        #从测试数据集里导入数据，测试准确率
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))
```

### 7、在简单的MNIST的基础上进行几处优化：

**a. 将二次代价函数换成交叉熵代价函数**

**b. 防止过拟合**

- ​     增加数据集

- ​     正则化    $C=C_0+\frac{\lambda}{2n}\sum\limits_{w}w^2$  其中$C_0$是原始的代价函数。

- ​     Dropout      训练的时候用部分神经元进行训练（可能每次训练，选择不同的神经元进行训练）               

​                                 测试的时候，用所有的神经元进行测试​   

 **c.使用不同的优化函数**

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

#每个批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义几个placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
#定义用于Dropout的概率
keep_prob=tf.placeholder(tf.float32)

#定义学习率
lr = tf.Variable(0.001, dtype=tf.float32)

#创建一个简单的神经网络,此处表示下一个隐藏层有2000个神经元
#原来的初始值时0，现在是从截断的正态分布中输出随机值
W1 = tf.Variable(tf.truncated_normal([784,2000],stddev=0.1))
b1 = tf.Variable(tf.zeros([2000])+0.1)
L1 = tf.nn.tanh(tf.matmul(x,W1)+b1)
L1_drop = tf.nn.dropout(L1,keep_prob) 

W2 = tf.Variable(tf.truncated_normal([2000,2000],stddev=0.1))
b2 = tf.Variable(tf.zeros([2000])+0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop,W2)+b2)
L2_drop = tf.nn.dropout(L2,keep_prob) 

W3 = tf.Variable(tf.truncated_normal([2000,1000],stddev=0.1))
b3 = tf.Variable(tf.zeros([1000])+0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop,W3)+b3)
L3_drop = tf.nn.dropout(L3,keep_prob) 

W4 = tf.Variable(tf.truncated_normal([1000,10],stddev=0.1))
b4 = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.nn.softmax(tf.matmul(L3_drop,W4)+b4)

#二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
#使用交叉熵代价函数
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

#使用梯度下降法
#train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#使用Adam优化器
train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#结果存放在一个布尔型列表中
#argmax返回一维张量中最大的值所在的位置
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(31):
        #每进行一个周期的训练，学习率降低
        sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
        for batch in range(n_batch):
            batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})
        
        learning_rate = sess.run(lr)
        #分别用训练集和测试集数据计算准确率
        #keep_prob的值表示有多少比例的神经元工作
        test_acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        train_acc = sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob:1.0})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(test_acc) +",Training Accuracy " + str(train_acc）+", Learning Rate= " + str(learning_rate))
```

​        **在程序的最后，分别用训练集和测试集数据计算准确率。当二者的准确率差别较大时，则可能是发生了过拟合。本例中，使用了dropout之后，训练的精确度和测试的精确度相差不大。**

### 8、使用Tensorboard进行可视化

**a.使用tensorboard查看网络结构**

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

#每个批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#命名空间，名字随便起
with tf.name_scope('input'):
    #定义两个placeholder
    x = tf.placeholder(tf.float32,[None,784],name='x_input')
    y = tf.placeholder(tf.float32,[None,10],name='y_input')

with tf.name_scope('layer'):
    #创建一个简单的神经网络
    with tf.name_scope('wights'):
    	W = tf.Variable(tf.zeros([784,10]))
    with tf.name_scope('biases'):
		b = tf.Variable(tf.zeros([10]))
    with tf.name_scope('wx_plus_b'):
    	wx_plus_b = tf.matmul(x,W) + b
    with tf.name_scope('softmax'):
    	prediction = tf.nn.softmax(wx_plus_b)

#二次代价函数
with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.square(y-prediction))
#使用梯度下降法
with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#结果存放在一个布尔型列表中
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
		correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
    	#argmax返回一维张量中最大的值所在的位置
    #求准确率
    with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    #设置tensorboard的保存路径，这里设为当前目录下logs文件夹内，没有的话，会新建一个
    writer = tf.summary.FileWriter('logs/',sess.graph)
    for epoch in range(1):
        for batch in range(n_batch):
            batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))
```

执行完此代码后，会在logs文件下生成一个文件，在命令窗口执行命令 ： **tensorboard --logdir=路径**

然后会生成一个网址，在浏览器（谷歌或者火狐）中打开这个网址。

要想执行新的图，先Ctrl+C两次，停止执行上一个，然后将原来logs中的文件删除，从新执行。

**b.查看网络运行时参数变化图**

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

#每个批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#参数概要，定义一个函数，来查看数据
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)#平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)#标准差
        tf.summary.scalar('max', tf.reduce_max(var))#最大值
        tf.summary.scalar('min', tf.reduce_min(var))#最小值
        tf.summary.histogram('histogram', var)#直方图

#命名空间
with tf.name_scope('input'):
    #定义两个placeholder
    x = tf.placeholder(tf.float32,[None,784],name='x-input')
    y = tf.placeholder(tf.float32,[None,10],name='y-input')
    
with tf.name_scope('layer'):
    #创建一个简单的神经网络
    with tf.name_scope('wights'):
        W = tf.Variable(tf.zeros([784,10]),name='W')
        variable_summaries(W)
    with tf.name_scope('biases'):    
        b = tf.Variable(tf.zeros([10]),name='b')
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x,W) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)

#二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    #使用梯度下降法
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        #结果存放在一个布尔型列表中
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax返回一维张量中最大的值所在的位置
    with tf.name_scope('accuracy'):
        #求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy',accuracy)
        
#合并所有的summary
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/',sess.graph)
    for epoch in range(51):
        for batch in range(n_batch):
            batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
            summary,_ = sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys})
        #这个是将每个周期的数据，作为一个样本点，打印出来，可以在tensorboard中看到
        #这里会打印出51个点
        writer.add_summary(summary,epoch)
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))

#使用这个程序，可以打印出2001个点
# for i in range(2001):
#     #m每个批次100个样本
#     batch_xs,batch_ys = mnist.train.next_batch(100)
#     summary,_ = sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys})
#     writer.add_summary(summary,i)
#     if i%500 == 0:
#         print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))
```

**c. tensorboard可视化**

```python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

#载入数据集
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
#运行次数
max_steps = 1001
#图片数量
image_num = 3000
#文件路径
DIR = "D:/Tensorflow/"

#定义会话
sess = tf.Session()

#载入图片,将0-image_num（3000）张图片打包
embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]), trainable=False, name='embedding')

#参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)#平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)#标准差
        tf.summary.scalar('max', tf.reduce_max(var))#最大值
        tf.summary.scalar('min', tf.reduce_min(var))#最小值
        tf.summary.histogram('histogram', var)#直方图

#命名空间
with tf.name_scope('input'):
    #这里的none表示第一个维度可以是任意的长度
    x = tf.placeholder(tf.float32,[None,784],name='x-input')
    #正确的标签
    y = tf.placeholder(tf.float32,[None,10],name='y-input')

#显示图片
with tf.name_scope('input_reshape'):
    #这里，-1代表一个不确定的值（x中的None），图片大小为28x28,灰度图片，维度为1
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

with tf.name_scope('layer'):
    #创建一个简单神经网络
    with tf.name_scope('weights'):
        W = tf.Variable(tf.zeros([784,10]),name='W')
        variable_summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]),name='b')
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x,W) + b
    with tf.name_scope('softmax'):    
        prediction = tf.nn.softmax(wx_plus_b)

with tf.name_scope('loss'):
    #交叉熵代价函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    #使用梯度下降法
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

#初始化变量
sess.run(tf.global_variables_initializer())

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        #结果存放在一个布尔型列表中
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax返回一维张量中最大的值所在的位置
    with tf.name_scope('accuracy'):
        #求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))#把correct_prediction变为float32类型
        tf.summary.scalar('accuracy',accuracy)

#产生metadata文件，这里是将3000张测试图片的标签存到这个文件中
if tf.gfile.Exists(DIR + 'projector/projector/metadata.tsv'):
    tf.gfile.DeleteRecursively(DIR + 'projector/projector/metadata.tsv')
with open(DIR + 'projector/projector/metadata.tsv', 'w') as f:
    labels = sess.run(tf.argmax(mnist.test.labels[:],1))
    for i in range(image_num):   
        f.write(str(labels[i]) + '\n')        
        
#合并所有的summary
merged = tf.summary.merge_all()   

#将图的结构存起来
projector_writer = tf.summary.FileWriter(DIR + 'projector/projector',sess.graph)
#将网络的模型保存起来
saver = tf.train.Saver()
#接下来一般是固定操作
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = embedding.name
embed.metadata_path = DIR + 'projector/projector/metadata.tsv'
embed.sprite.image_path = DIR + 'projector/data/mnist_10k_sprite.png'
#将data文件夹下的大图，切分成28x28的小图片
embed.sprite.single_image_dim.extend([28,28])
projector.visualize_embeddings(projector_writer,config)

#每次训练100张图片，训练1000次
for i in range(max_steps):
    #每个批次100个样本
    batch_xs,batch_ys = mnist.train.next_batch(100)
    #一般是固定搭配
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    
    summary,_ = sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys},options=run_options,run_metadata=run_metadata)
    projector_writer.add_run_metadata(run_metadata, 'step%03d' % i)
    projector_writer.add_summary(summary, i)
    
    if i%100 == 0:
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print ("Iter " + str(i) + ", Testing Accuracy= " + str(acc))
#将训练好的模型保存到对应路径
saver.save(sess, DIR + 'projector/projector/a_model.ckpt', global_step=max_steps)
projector_writer.close()
sess.close()
```

### 9、卷积神经网络CNN

**传统神经网络存在的问题：**

权值太多，需要大量的样本进行训练，计算量太大。

CNN通过感受野和权值共享减少了神经网络需要训练的参数。

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#每个批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)#平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)#标准差
        tf.summary.scalar('max', tf.reduce_max(var))#最大值
        tf.summary.scalar('min', tf.reduce_min(var))#最小值
        tf.summary.histogram('histogram', var)#直方图

#初始化权值
def weight_variable(shape,name):
    initial = tf.truncated_normal(shape,stddev=0.1)#生成一个截断的正态分布
    return tf.Variable(initial,name=name)

#初始化偏置
def bias_variable(shape,name):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial,name=name)

#卷积层
def conv2d(x,W):
    #x input tensor of shape `[batch, in_height, in_width, in_channels]`
    #W filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
    #`strides[0] = strides[3] = 1`. strides[1]代表x方向的步长，strides[2]代表y方向的步长
    #padding: A `string` from: `"SAME", "VALID"`
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#池化层
def max_pool_2x2(x):
    #ksize [1,x,y,1],池化窗口的大小
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#命名空间
with tf.name_scope('input'):
    #定义两个placeholder
    x = tf.placeholder(tf.float32,[None,784],name='x-input')
    y = tf.placeholder(tf.float32,[None,10],name='y-input')
    with tf.name_scope('x_image'):
        #改变x的格式转为4D的向量[batch, in_height, in_width, in_channels]`
        x_image = tf.reshape(x,[-1,28,28,1],name='x_image')


with tf.name_scope('Conv1'):
    #初始化第一个卷积层的权值和偏置
    with tf.name_scope('W_conv1'):
        W_conv1 = weight_variable([5,5,1,32],name='W_conv1')
        #5*5的采样窗口，32个卷积核从1个平面抽取特征（输入一个平面），经过此卷积层之后，会有32个特征平面
    with tf.name_scope('b_conv1'):  
        b_conv1 = bias_variable([32],name='b_conv1')
        #每一个卷积核一个偏置值

    #把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    with tf.name_scope('conv2d_1'):
        conv2d_1 = conv2d(x_image,W_conv1) + b_conv1
    with tf.name_scope('relu'):
        h_conv1 = tf.nn.relu(conv2d_1)
    with tf.name_scope('h_pool1'):
        h_pool1 = max_pool_2x2(h_conv1)#进行max-pooling

with tf.name_scope('Conv2'):
    #初始化第二个卷积层的权值和偏置
    with tf.name_scope('W_conv2'):
        W_conv2 = weight_variable([5,5,32,64],name='W_conv2')
        #5*5的采样窗口，64个卷积核从32个平面抽取特征（此时的输入有32个平面），输出64个平面
    with tf.name_scope('b_conv2'):  
        b_conv2 = bias_variable([64],name='b_conv2')
        #每一个卷积核一个偏置值

    #把h_pool1和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    with tf.name_scope('conv2d_2'):
        conv2d_2 = conv2d(h_pool1,W_conv2) + b_conv2
    with tf.name_scope('relu'):
        h_conv2 = tf.nn.relu(conv2d_2)
    with tf.name_scope('h_pool2'):
        h_pool2 = max_pool_2x2(h_conv2)#进行max-pooling

#28*28的图片第一次卷积后还是28*28，第一次池化后变为14*14
#第二次卷积后为14*14，第二次池化后变为了7*7
#进过上面操作后得到64张7*7的平面

with tf.name_scope('fc1'):
    #初始化第一个全连接层的权值
    with tf.name_scope('W_fc1'):
        W_fc1 = weight_variable([7*7*64,1024],name='W_fc1')
        #上一场有7*7*64个神经元，全连接层有1024个神经元
    with tf.name_scope('b_fc1'):
        b_fc1 = bias_variable([1024],name='b_fc1')#1024个节点

    #把池化层2的输出扁平化为1维
    with tf.name_scope('h_pool2_flat'):
        h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64],name='h_pool2_flat')
    #求第一个全连接层的输出
    with tf.name_scope('wx_plus_b1'):
        wx_plus_b1 = tf.matmul(h_pool2_flat,W_fc1) + b_fc1
    with tf.name_scope('relu'):
        h_fc1 = tf.nn.relu(wx_plus_b1)

    #keep_prob用来表示神经元的输出概率
    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    with tf.name_scope('h_fc1_drop'):
        h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob,name='h_fc1_drop')

with tf.name_scope('fc2'):
    #初始化第二个全连接层
    with tf.name_scope('W_fc2'):
        W_fc2 = weight_variable([1024,10],name='W_fc2')
    with tf.name_scope('b_fc2'):    
        b_fc2 = bias_variable([10],name='b_fc2')
    with tf.name_scope('wx_plus_b2'):
        wx_plus_b2 = tf.matmul(h_fc1_drop,W_fc2) + b_fc2
    with tf.name_scope('softmax'):
        #计算输出
        prediction = tf.nn.softmax(wx_plus_b2)

#交叉熵代价函数
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction),name='cross_entropy')
    tf.summary.scalar('cross_entropy',cross_entropy)
    
#使用AdamOptimizer进行优化
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#求准确率
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        #结果存放在一个布尔列表中
        correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))#argmax返回一维张量中最大的值所在的位置
    with tf.name_scope('accuracy'):
        #求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy',accuracy)
        
#合并所有的summary
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('logs/train',sess.graph)
    test_writer = tf.summary.FileWriter('logs/test',sess.graph)
    for i in range(1001):
        #训练模型
        batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
        sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.5})
        #记录训练集计算的参数
        summary = sess.run(merged,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
        train_writer.add_summary(summary,i)
        #记录测试集计算的参数
        batch_xs,batch_ys =  mnist.test.next_batch(batch_size)
        summary = sess.run(merged,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
        test_writer.add_summary(summary,i)
    
        if i%100==0:
            test_acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
            train_acc = sess.run(accuracy,feed_dict={x:mnist.train.images[:10000],y:mnist.train.labels[:10000],keep_prob:1.0})
            print ("Iter " + str(i) + ", Testing Accuracy= " + str(test_acc) + ", Training Accuracy= " + str(train_acc))
```

**MNIST的AlexNet实现**

```python
# 贡献者：{吴翔 QQ：99456786}
# tensorflow版本基于0.12
# 源代码出处：
# 数据集下载地址：{http://yann.lecun.com/exdb/mnist/}
# 数据集下载到本地后存储的路径："D:\tensorflow\Data_sets\MNIST_data"
'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
'''

import tensorflow as tf

# 输入数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("D:\\tensorflow\\Data_sets\\MNIST_data", one_hot=True)

# 定义网络的超参数
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 5

# 定义网络的参数
n_input = 784 # 输入的维度 (img shape: 28*28)
n_classes = 10 # 标记的维度 (0-9 digits)
dropout = 0.75 # Dropout的概率，输出的可能性

# 输入占位符
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# 定义卷积操作
def conv2d(name,x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x,name=name)  # 使用relu激活函数

# 定义池化层操作
def maxpool2d(name,x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME',name=name)

# 规范化操作
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0,
                     beta=0.75, name=name)

# 定义所有的网络参数
weights = {
    'wc1': tf.Variable(tf.random_normal([11, 11, 1, 96])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
    'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
    'wd1': tf.Variable(tf.random_normal([4*4*256, 4096])),
    'wd2': tf.Variable(tf.random_normal([4096, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([96])),
    'bc2': tf.Variable(tf.random_normal([256])),
    'bc3': tf.Variable(tf.random_normal([384])),
    'bc4': tf.Variable(tf.random_normal([384])),
    'bc5': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([4096])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# 定义整个网络
def alex_net(x, weights, biases, dropout):
    # 向量转为矩阵 Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # 第一层卷积
    # 卷积
    conv1 = conv2d('conv1', x, weights['wc1'], biases['bc1'])
    # 下采样
    pool1 = maxpool2d('pool1', conv1, k=2)
    # 规范化
    norm1 = norm('norm1', pool1, lsize=4)

    # 第二层卷积
    # 卷积
    conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
    # 最大池化（向下采样）
    pool2 = maxpool2d('pool2', conv2, k=2)
    # 规范化
    norm2 = norm('norm2', pool2, lsize=4)

    # 第三层卷积
    # 卷积
    conv3 = conv2d('conv3', norm2, weights['wc3'], biases['bc3'])
    # 规范化
    norm3 = norm('norm3', conv3, lsize=4)

    # 第四层卷积
    conv4 = conv2d('conv4', norm3, weights['wc4'], biases['bc4'])

    # 第五层卷积
    conv5 = conv2d('conv5', conv4, weights['wc5'], biases['bc5'])
    # 最大池化（向下采样）
    pool5 = maxpool2d('pool5', conv5, k=2)
    # 规范化
    norm5 = norm('norm5', pool5, lsize=4)


    # 全连接层1
    fc1 = tf.reshape(norm5, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 =tf.add(tf.matmul(fc1, weights['wd1']),biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # dropout
    fc1=tf.nn.dropout(fc1,dropout)

    # 全连接层2
    fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
    fc2 =tf.add(tf.matmul(fc2, weights['wd2']),biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    # dropout
    fc2=tf.nn.dropout(fc2,dropout)

    # 输出层
    out = tf.add(tf.matmul(fc2, weights['out']) ,biases['out'])
    return out


# 构建模型
pred = alex_net(x, weights, biases, keep_prob)

# 定义损失函数和优化器
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 评估函数
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化变量
init = tf.global_variables_initializer()

# 开启一个训练
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # 开始训练，直到达到training_iters，即200000
    while step * batch_size < training_iters:
        #获取批量数据
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        if step % display_step == 0:
            # 计算损失值和准确度，输出
            loss,acc = sess.run([cost,accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print ("Optimization Finished!")
    # 计算测试集的精确度
    print ("Testing Accuracy:",
           sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                         y: mnist.test.labels[:256],
                                         keep_prob: 1.}))
```



### 10、递归神经网络RNN

- **RNN（Recurrent Neural Network）**

![](/RNN1.jpg)



![](/RNN2.jpg)

**上一层的输出，参与到下一层的输入。**



- **LSTM（Long Short Term Memory）**

​    目的是，将以前层输出的值，有用的留下，无用的抛弃。

通过，Input Gate,Forget Gate,Output Gate,控制信号的传递。

Input Gate控制是否允许信号输入。

Forget Gate控制信号向下一层传递的情况，是否忘记这个信号。

Output Gate控制信号是否输出。

在LSTM中，上一层的输出，是否要向后传，传多少，都是可以控制的。



![](/LSTM.jpg) ![](/LSTM1.jpg) 

$i_t$表示Input Gate的输出，$f_t$表示Forget Gate的输出，$o_t$表示Output Gate的输出；

$x_t$表示输入，$h_t$表示最后的输出，$h_{t-1}​$表示上一层输出

$\hat{c_t}​$表示最下面的输入单元的输出，$c_t​$表示中间的Cell单元的输出

W、U、b是参数

在函数tf.nn.dynamic_rnn的返回值的state中，cell state 等于这里的$c_t$, hidden state 等于这里的$h_t$



![](/LSTM2.jpg)

![](/LSTM3.jpg)

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# 输入图片是28*28
n_inputs = 28 #每次输入图像的一行，一行有28个数据，这里是序列中的一个点
max_time = 28 #一共28行，在这里代表了序列的长度
lstm_size = 100 #隐层单元，这里不叫神经元，每个单元称为block
n_classes = 10 # 10个分类
batch_size = 50 #每批次50个样本
n_batch = mnist.train.num_examples // batch_size #计算一共有多少个批次

#这里的none表示第一个维度可以是任意的长度,在这里等于batch_size
x = tf.placeholder(tf.float32,[None,784])
#正确的标签
y = tf.placeholder(tf.float32,[None,10])

#初始化权值
weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))
#初始化偏置值
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))


#定义RNN网络
def RNN(X,weights,biases):
    # inputs=[batch_size, max_time, n_inputs]
    inputs = tf.reshape(X,[-1,max_time,n_inputs])
    #定义LSTM基本CELL,即BasicLSTMCell
    lstm_cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(lstm_size)
    # final_state[0]是cell state
    # final_state[1]是hidden_state
    outputs,final_state = tf.nn.dynamic_rnn(lstm_cell,inputs,dtype=tf.float32)
    #这里的final_state[1]表示整个序列所有点的最终输出
    results = tf.nn.softmax(tf.matmul(final_state[1],weights) + biases)
    return results
    
    
#计算RNN的返回结果
prediction= RNN(x, weights, biases)  
#损失函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
#使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax返回一维张量中最大的值所在的位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))#把correct_prediction变为float32类型
#初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(6):
        for batch in range(n_batch):
            batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print ("Iter " + str(epoch) + ", Testing Accuracy= " + str(acc))
```

### 11、保存和载入模型

**（1）保存模型**

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

#每个批次100张照片
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#创建一个简单的神经网络，输入层784个神经元，输出层10个神经元
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

#二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax返回一维张量中最大的值所在的位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#保存和载入模型，都要用到
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(11):
        for batch in range(n_batch):
            batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))
    #保存模型
    saver.save(sess,'net/my_net.ckpt')
```

**（2）载入模型**

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

#每个批次100张照片
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#创建一个简单的神经网络，输入层784个神经元，输出层10个神经元
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

#二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax返回一维张量中最大的值所在的位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    #这里，先用初始化的参数计算一次准确率，再用载入的模型计算一次，二者对比
    sess.run(init)
    print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))
    saver.restore(sess,'net/my_net.ckpt')
    print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))
```

### 12、下载inception-v3网络并用tensorboard查看其结构

```python
import tensorflow as tf
import os
import tarfile
import requests

#inception模型下载地址
inception_pretrain_model_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

#模型存放地址
inception_pretrain_model_dir = "inception_model"
if not os.path.exists(inception_pretrain_model_dir):
    os.makedirs(inception_pretrain_model_dir)
    
#获取文件名，以及文件路径
filename = inception_pretrain_model_url.split('/')[-1]
filepath = os.path.join(inception_pretrain_model_dir, filename)

#下载模型
if not os.path.exists(filepath):
    print("download: ", filename)
    r = requests.get(inception_pretrain_model_url, stream=True)
    with open(filepath, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
print("finish: ", filename)
#解压文件
tarfile.open(filepath, 'r:gz').extractall(inception_pretrain_model_dir)
 
#模型结构存放文件
log_dir = 'inception_log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

#classify_image_graph_def.pb为google训练好的模型
inception_graph_def_file = os.path.join(inception_pretrain_model_dir, 'classify_image_graph_def.pb')
with tf.Session() as sess:
    #创建一个图来存放google训练好的模型
    with tf.gfile.FastGFile(inception_graph_def_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    #保存图的结构
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    writer.close()
```

### 13、使用inception-v3网络进行各种图像的识别

```python
import tensorflow as tf
import os
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt

class NodeLookup(object):
    def __init__(self):
        #这两个数据说明，此网络可以识别1000类目标
        label_lookup_path = 'inception_model/imagenet_2012_challenge_label_map_proto.pbtxt'   
        uid_lookup_path = 'inception_model/imagenet_synset_to_human_label_map.txt'
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        # 加载分类字符串n********对应分类名称的文件
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        #一行一行读取数据
        for line in proto_as_ascii_lines :
            #去掉换行符
            line=line.strip('\n')
            #按照'\t'分割
            parsed_items = line.split('\t')
            #获取分类编号
            uid = parsed_items[0]
            #获取分类名称
            human_string = parsed_items[1]
            #保存编号字符串n********与分类名称映射关系
            uid_to_human[uid] = human_string

        # 加载分类字符串n********对应分类编号1-1000的文件
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        node_id_to_uid = {}
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                #获取分类编号1-1000
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                #获取编号字符串n********
                target_class_string = line.split(': ')[1]
                #保存分类编号1-1000与编号字符串n********映射关系
                node_id_to_uid[target_class] = target_class_string[1:-2]

        #建立分类编号1-1000对应分类名称的映射关系
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            #获取分类名称
            name = uid_to_human[val]
            #建立分类编号1-1000到分类名称的映射关系
            node_id_to_name[key] = name
        return node_id_to_name

    #传入分类编号1-1000返回分类名称，即网络输出的是分类编号，通过此函数，转化成分类名称
    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


#创建一个图来存放google训练好的模型
with tf.gfile.FastGFile('inception_model/classify_image_graph_def.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    #遍历目录
    for root,dirs,files in os.walk('images/'):
        for file in files:
            #载入图片
            image_data = tf.gfile.FastGFile(os.path.join(root,file), 'rb').read()
            predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})#图片格式是jpg格式
            predictions = np.squeeze(predictions)#把结果转为1维数据

            #打印图片路径及名称
            image_path = os.path.join(root,file)
            print(image_path)
            #显示图片
            img=Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()

            #排序，按照不同类别的概率大小
            top_k = predictions.argsort()[-5:][::-1]
            node_lookup = NodeLookup()
            for node_id in top_k:     
                #通过输出的类别的编号，获取分类名称
                human_string = node_lookup.id_to_string(node_id)
                #获取该分类的置信度
                score = predictions[node_id]
                print('%s (score = %.5f)' % (human_string, score))
            print()
```

### 14、在原有模型基础上重新训练自己的模型（迁移学习）

此方法，是在已经训练好的模型的基础上，加上新的全连接和分类函数，然后重新训练。

（先在github上下载源码）

- 训练代码来自

G:\关于tensorflow\tensorflow源码\tensorflow-master\tensorflow-master\tensorflow\examples\image_retraining

- 数据集下载地址：

http://www.robots.ox.ac.uk/~vgg/data/

注：程序的所需代码和文件都在retrain目录中

- 如果是windows系统，直接执行批处理文件retrain.bat即可，注意改里面的路径。

- 如果是linux系统，执行下面的命令：

python3 retrain.py --image_dir ./train/ --bottleneck_dir ./bottleneck --how_many_training_steps 200 **--model_dir ./inception_model** --output_graph output_graph.pb --output_labels output_labels.txt

注：这个官方程序与前几年相比有改动，执行时，会自动去网络上下载最初的模型。即上面加粗的用来引入旧模型的命令，不在适合现在的程序。查看命令的全部列表：**python retrain.py -h**

执行此程序的官方教程链接：https://www.tensorflow.org/tutorials/image_retraining



**接下来使用新训练好的模型，识别图片：（与上一节用inception-v3识别图片类似）**

```python
import tensorflow as tf
import os
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt

lines = tf.gfile.GFile('retrain/output_labels.txt').readlines()
uid_to_human = {}
#一行一行读取数据
for uid,line in enumerate(lines) :
    #去掉换行符
    line=line.strip('\n')
    uid_to_human[uid] = line

def id_to_string(node_id):
    if node_id not in uid_to_human:
        return ''
    return uid_to_human[node_id]


#创建一个图来存放google训练好的模型
with tf.gfile.FastGFile('retrain/output_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    #遍历目录
    for root,dirs,files in os.walk('retrain/images/'):
        for file in files:
            #载入图片
            image_data = tf.gfile.FastGFile(os.path.join(root,file), 'rb').read()
            predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})#图片格式是jpg格式
            predictions = np.squeeze(predictions)#把结果转为1维数据

            #打印图片路径及名称
            image_path = os.path.join(root,file)
            print(image_path)
            #显示图片
            img=Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()

            #排序
            top_k = predictions.argsort()[::-1]
            print(top_k)
            for node_id in top_k:     
                #获取分类名称
                human_string = id_to_string(node_id)
                #获取该分类的置信度
                score = predictions[node_id]
                print('%s (score = %.5f)' % (human_string, score))
            print()
```

### 15、从头开始，训练新的模型

先在github上下载源码，TensorFlow/**model**,本次用其中的slim,其路径为：models-master\research

将slim文件夹复制出来使用。

在slim下新建两个文件夹，images和model，前者用来存放数据集，包括原始图片和转化之后的tfrecord文件，后者用来存放训练好的模型。

在datasets文件夹中，添加程序myimages.py，用来读取tfrecord文件，另外修改dataset_factory.py，在第32行，添加自己的数据集。

**首先，需要先将原始图片转化成tfrecord格式**

```python
import tensorflow as tf
import os
import random
import math
import sys

#验证集数量
_NUM_TEST = 500
#随机种子
_RANDOM_SEED = 0
#数据块
_NUM_SHARDS = 5
#数据集路径
DATASET_DIR = "./slim/images/"
#标签文件名字
LABELS_FILENAME = "./labels.txt"

#将图片文件转化成tfrecord文件，这种格式的底层为protobuf
#定义tfrecord文件的路径+名字
def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = 'image_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)

#判断tfrecord文件是否存在
def _dataset_exists(dataset_dir):
    for split_name in ['train', 'test']:
        for shard_id in range(_NUM_SHARDS):
            #定义tfrecord文件的路径+名字
            output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)
        if not tf.gfile.Exists(output_filename):
            return False
    return True

#获取所有文件以及分类，（几个数据集文件夹的名字就是分类名）
def _get_filenames_and_classes(dataset_dir):
    #数据目录
    directories = []
    #分类名称
    class_names = []
    for filename in os.listdir(dataset_dir):
        #合并文件路径
        path = os.path.join(dataset_dir, filename)
        #判断该路径是否为目录
        if os.path.isdir(path):
            #加入数据目录
            directories.append(path)
            #加入类别名称
            class_names.append(filename)

    photo_filenames = []
    #循环每个分类的文件夹
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            #把图片加入图片列表
            photo_filenames.append(path)

    return photo_filenames, class_names

def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def image_to_tfexample(image_data, image_format, class_id):
    #Abstract base class for protocol messages.
    return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
    }))

def write_label_file(labels_to_class_names, dataset_dir,filename=LABELS_FILENAME):
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))

#把数据转为TFRecord格式
def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
    assert split_name in ['train', 'test']
    #计算每个数据块有多少数据（当数据集数量很多时，需要分块）
    num_per_shard = int(len(filenames) / _NUM_SHARDS)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            for shard_id in range(_NUM_SHARDS):
                #定义tfrecord文件的路径+名字
                output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)
                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    #每一个数据块开始的位置
                    start_ndx = shard_id * num_per_shard
                    #每一个数据块最后的位置
                    end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        try:
                            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i+1, len(filenames), shard_id))
                            sys.stdout.flush()
                            #读取图片
                            image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
                            #获得图片的类别名称
                            class_name = os.path.basename(os.path.dirname(filenames[i]))
                            #找到类别名称对应的id
                            class_id = class_names_to_ids[class_name]
                            #生成tfrecord文件
                            example = image_to_tfexample(image_data, b'jpg', class_id)
                            tfrecord_writer.write(example.SerializeToString())
                        except IOError as e:
                            print("Could not read:",filenames[i])
                            print("Error:",e)
                            print("Skip it\n")
                            
    sys.stdout.write('\n')
    sys.stdout.flush()


if __name__ == '__main__':
    #判断tfrecord文件是否存在
    if _dataset_exists(DATASET_DIR):
        print('tfcecord文件已存在')
    else:
        #获得所有图片以及分类
        photo_filenames, class_names = _get_filenames_and_classes(DATASET_DIR)
        #把分类转为字典格式，类似于{'house': 3, 'flower': 1, 'plane': 4, 'guitar': 2, 'animal': 0}
        class_names_to_ids = dict(zip(class_names, range(len(class_names))))

        #把数据切分为训练集和测试集
        random.seed(_RANDOM_SEED)
        random.shuffle(photo_filenames)
        training_filenames = photo_filenames[_NUM_TEST:]
        testing_filenames = photo_filenames[:_NUM_TEST]

        #数据转换
        _convert_dataset('train', training_filenames, class_names_to_ids, DATASET_DIR)
        _convert_dataset('test', testing_filenames, class_names_to_ids, DATASET_DIR)

        #输出labels文件
        labels_to_class_names = dict(zip(range(len(class_names)), class_names))
        write_label_file(labels_to_class_names, DATASET_DIR)
```

（注：这个程序，虚拟机没有运行成功，在服务器上成功运行，注意路径）

进入slim文件夹，执行下面的命令。开始训练

python3.5 train_image_classifier.py --train_dir ./model --dataset_name myimages --dataset_split_name train --dataset_dir ./images --max_number_of_steps 1000 --model_name inception_v3 --batch_size 10



## 四、TensorFlow 的API索引

**注：在jupyter notebook中，按Shift+Tab键，可以查看各种函数方法的详解**

------



### （1）算术操作

##### <1>tf.add

tf.add(x, y, name=None)      求和

##### <2>tf.sub

tf.sub(x, y, name=None)       减法

##### <3>tf.mul

tf.mul(x, y, name=None)       乘法

##### <4>tf.div

tf.div(x, y, name=None)         除法

##### <5>tf.mod

tf.mod(x, y, name=None)        取模

##### <6>tf.abs

tf.abs(x, name=None)             求绝对值

##### <7>tf.neg

tf.neg(x, name=None)           取负 (y= -x)

##### <8>tf.sign

tf.sign(x, name=None)           返回符号 y= sign(x) = -1 if x < 0; 0 if x == 0; 1 if x > 0

##### <9>tf.inv

tf.inv(x, name=None)              取反

##### <10>tf.square

tf.square(x, name=None)        计算平方 (y= x * x = x^2)

##### <11>tf.round

tf.round(x, name=None)         

舍入最接近的整数

\# ‘a’ is [0.9, 2.5, 2.3, -4.4]

tf.round(a) ==> [ 1.0, 3.0, 2.0, -4.0 ]

##### <12>tf.sqrt

tf.sqrt(x, name=None)               开根号 (y= \sqrt{x} = x^{1/2})

##### <13>tf.pow

tf.pow(x, y, name=None)

幂次方 

\# tensor ‘x’ is [[2, 2], [3, 3]]

\# tensor ‘y’ is [[8, 16], [2, 3]]

tf.pow(x, y) ==> [[256, 65536], [9, 27]]

##### <14>tf.exp

tf.exp(x, name=None)         计算e的次方

##### <15>tf.log

tf.log(x, name=None)            计算log，一个输入计算e的ln，两输入以第二输入为底

##### <16>tf.maximum

tf.maximum(x, y, name=None)             返回最大值 (x>y ? x : y)

##### <17>tf.minimum

tf.minimum(x, y, name=None)              返回最小值 (x< y ? x : y)

##### <18>tf.cos

tf.cos(x, name=None)                       三角函数cosine

##### <19>tf.sin

tf.sin(x, name=None)                        三角函数sine

##### <20>tf.tan

tf.tan(x, name=None)                       三角函数tan

##### <21>tf.atan

tf.atan(x, name=None)                     三角函数ctan

### （2）张量操作

#### 1、常用操作

##### <1>tf.Variable\tf.get_variable

**a、图变量的初始化方法**

对于一般的Python代码，变量的初始化就是变量的定义，向下面这样：

 In [1]: x = 3

In [2]: y = 3 * 5

In [3]: y

Out[3]: 15

如果我们模仿上面的写法来进行TensorFlow编程，就会出现下面的”怪现象”：

In [1]: import tensorflow as tf

In [2]: x = tf.Variable(3, name='x')

In [3]: y = x * 5

In [4]: print(y)

Tensor("mul:0", shape=(), dtype=int32)

y的值并不是我们预想中的15，而是一个莫名其妙的输出——”



In [1]: import tensorflow as tf

In [2]: x = tf.Variable(3, name='x')

In [3]: y = x * 5

In [4]: sess = tf.InteractiveSession()

In [5]: sess.run(tf.global_variables_initializer())

In [6]: sess.run(y)

Out[6]: 15

在TensorFlow的世界里，变量的定义和初始化是分开的，所有关于图变量的赋值和计算都要通过tf.Session的run来进行。想要将所有图变量进行集体初始化时应该使用tf.global_variables_initializer。

**b、两种定义图变量的方法**

**<1>tf.Variable**

tf.Variable.init(initial_value, trainable=True, collections=None, validate_shape=True, name=None)

参数名称            参数类型                                                含义

initial_value      所有可以转换为Tensor的类型     变量的初始值

trainable          bool     如果为True，会把它加入到GraphKeys.TRAINABLE_VARIABLES，才能对它使用Optimizer

collections               list                                        指定该图变量的类型、默认为[GraphKeys.GLOBAL_VARIABLES]

validate_shape       bool                                     如果为False，则不进行类型和维度检查

name                        string                                   变量的名称，如果没有指定则系统会自动分配一个唯一的值

**虽然有一堆参数，但只有第一个参数initial_value是必需的**，用法如下（assign函数用于给图变量赋值）：

In [1]: import tensorflow as tf

In [2]: v = tf.Variable(3, name='v')

In [3]: v2 = v.assign(5)

In [4]: sess = tf.InteractiveSession()

In [5]: sess.run(v.initializer)

In [6]: sess.run(v)

Out[6]: 3

In [7]: sess.run(v2)

Out[7]: 5

**<2>tf.get_variable**

tf.get_variable跟tf.Variable都可以用来定义图变量，但是前者的必需参数（即第一个参数）并不是图变量的初始值，而是图变量的名称。

In [1]: import tensorflow as tf

In [2]: init = tf.constant_initializer([5])

In [3]: x = tf.get_variable('x', shape=[1], initializer=init)

In [4]: sess = tf.InteractiveSession()

In [5]: sess.run(x.initializer)

In [6]: sess.run(x)

Out[6]: array([ 5.], dtype=float32)

**3、scope如何划分命名空间**

一个深度学习模型的参数变量往往是成千上万的，不加上命名空间加以分组整理，将会成为可怕的灾难。TensorFlow的命名空间分为两种，tf.variable_scope和tf.name_scope。

下面示范使用tf.variable_scope把图变量划分为4组：

for i in range(4):

​    with tf.variable_scope('scope-{}'.format(i)):

​        for j in range(25):

​              v = tf.Variable(1, name=str(j))

可视化输出的结果如下：

![](/../typora%E5%9B%BE%E7%89%87%E6%80%BB/scope%E5%91%BD%E5%90%8D%E7%A9%BA%E9%97%B4.jpg)

**下面让我们来分析tf.variable_scope和tf.name_scope的区别**：

**<1>tf.variable_scope**

1、当使用tf.get_variable定义变量时，如果出现同名的情况将会引起报错 

In [1]: import tensorflow as tf

In [2]: with tf.variable_scope('scope'):

   ...:     v1 = tf.get_variable('var', [1])

   ...:     v2 = tf.get_variable('var', [1])

ValueError: Variable scope/var already exists, disallowed. Did you mean to set reuse=True in VarScope? Originally defined at:

2、而对于tf.Variable来说，却可以定义“同名”变量

In [1]: import tensorflow as tf

In [2]: with tf.variable_scope('scope'):

   ...:     v1 = tf.Variable(1, name='var')

   ...:     v2 = tf.Variable(2, name='var')

   ...:

In [3]: v1.name, v2.name

Out[3]: ('scope/var:0', 'scope/var_1:0')

但是把这些图变量的name属性打印出来，就可以发现它们的名称并不是一样的。

如果想使用tf.get_variable来定义另一个同名图变量，可以考虑加入新一层scope，比如：

In [1]: import tensorflow as tf

In [2]: with tf.variable_scope('scope1'):

   ...:     v1 = tf.get_variable('var', shape=[1])

   ...:     with tf.variable_scope('scope2'):

   ...:         v2 = tf.get_variable('var', shape=[1])

   ...:

In [3]: v1.name, v2.name

Out[3]: ('scope1/var:0', 'scope1/scope2/var:0')

**<2>tf.name_scope**

当tf.get_variable遇上tf.name_scope，它定义的变量的最终完整名称将不受这个tf.name_scope的影响，如下：

In [1]: import tensorflow as tf

In [2]: with tf.variable_scope('v_scope'):

   ...:     with tf.name_scope('n_scope'):

   ...:         x = tf.Variable([1], name='x')

   ...:         y = tf.get_variable('x', shape=[1], dtype=tf.int32)

   ...:         z = x + y

   ...:

In [3]: x.name, y.name, z.name

Out[3]: ('v_scope/n_scope/x:0', 'v_scope/x:0', 'v_scope/n_scope/add:0')

**4、图变量的复用**

想象一下，如果我们正在定义一个循环神经网络RNN，想复用上一层的参数以提高模型最终的表现效果，应该怎么做呢？

**<1>做法一：**

In [1]: import tensorflow as tf

In [2]: with tf.variable_scope('scope'):

   ...:     v1 = tf.get_variable('var', [1])

   ...:     tf.get_variable_scope().reuse_variables()

   ...:     v2 = tf.get_variable('var', [1])

   ...:

In [3]: v1.name, v2.name

Out[3]: ('scope/var:0', 'scope/var:0')

**<2> 做法二：**

In [1]: import tensorflow as tf

In [2]: with tf.variable_scope('scope'):

   ...:     v1 = tf.get_variable('x', [1])

   ...:

In [3]: with tf.variable_scope('scope', reuse=True):

   ...:     v2 = tf.get_variable('x', [1])

   ...:

In [4]: v1.name, v2.name

Out[4]: ('scope/x:0', 'scope/x:0')

**5、图变量的种类**

TensorFlow的图变量分为两类：**local_variables**和**global_variables**。

如果我们想定义一个不需要长期保存的临时图变量，可以向下面这样定义它：

with tf.name_scope("increment"):

​    zero64 = tf.constant(0, dtype=tf.int64)

​    current = tf.Variable(zero64, name="incr", trainable=False, collections=[ops.GraphKeys.LOCAL_VARIABLES])

------



##### <2>tf.truncated_normal

tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)

**参数:**

​        **shape:** 一维的张量，也是输出的张量。

​        **mean:** 正态分布的均值。

​        **stddev:** 正态分布的标准差。

​        **dtype:** 输出的类型。

​        **seed:** 一个整数，当设置之后，每次生成的随机数都一样。

​        **name:** 操作的名字。

从**截断**的正态分布中输出随机值。

生成的值服从具有指定平均值和标准偏差的正态分布，如果生成的值大于平均值两个标准偏差，则丢弃该值重新选择。

![](/../typora%E5%9B%BE%E7%89%87%E6%80%BB/%E6%AD%A3%E6%80%81%E5%88%86%E5%B8%83%E5%9B%BE.jpg)

1、μ是正态分布的位置参数，描述正态分布的[集中趋势](https://baike.baidu.com/item/%E9%9B%86%E4%B8%AD%E8%B6%8B%E5%8A%BF)位置。概率规律为取与μ邻近的值的概率大。正态分布以X=μ为[对称轴](https://baike.baidu.com/item/%E5%AF%B9%E7%A7%B0%E8%BD%B4)，左右完全对称。正态分布的期望、[均数](https://baike.baidu.com/item/%E5%9D%87%E6%95%B0)、[中位数](https://baike.baidu.com/item/%E4%B8%AD%E4%BD%8D%E6%95%B0)、众数相同，均等于μ。

2、σ描述正态分布资料数据分布的离散程度，σ越大，数据分布越分散，σ越小，数据分布越集中。也称为是正态分布的形状参数，σ越大，曲线越扁平，反之，σ越小，曲线越瘦高。

3、在正态分布的曲线中，横轴区间（μ-σ，μ+σ）内的面积为68.268949%。 横轴区间（μ-2σ，μ+2σ）内的面积为95.449974%。 横轴区间（μ-3σ，μ+3σ）内的面积为99.730020%。 
X落在（μ-3σ，μ+3σ）以外的概率小于千分之三，在实际问题中常认为相应的事件是不会发生的，基本上可以把区间（μ-3σ，μ+3σ）看作是随机变量X实际可能的取值区间，这称之为正态分布的“3σ”原则。 

**在tf.truncated_normal中如果x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择。这样保证了生成的值都在均值附近**。

------



##### <3>tf.random_normal

tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)

**从正态分布中输出随机值**

**参数:**

​    **shape:** 一维的张量，也是输出的张量。

​    **mean:** 正态分布的均值。

​    **stddev:** 正态分布的标准差。

​    **dtype:** 输出的类型。

​    **seed:** 一个整数，当设置之后，每次生成的随机数都一样。

​    **name:** 操作的名字。

##### <4>tf.constant

tf.constant( value, dtype=None ,shape=None, name='Const', verify_shape=False)

用来创建常量

**参数：**除了第一个value别的都不是必须的

**value:** 这个参数是必须的，可以是一个数值，也可以是一个列表。

​            1、 创建一个数值：tensor=tf.constant(1)

​            2、创建一个列表：tensor=tf.constant([1, 2])

​            为查看结果必须创建一个会话，并用取值函数eval()来查看创建的tensor的值：

​               sess=tf.Session()
​               with sess.as_default():
​                       print('结果是：', tensor.eval()) 

​               结果是：1   或者   结果是：[1 2]

**dtype:** 这个参数表示数据类型，一般可以是tf.float32,tf.float64等：

**shape:** 表示张量的形状，即维数以及每一维的大小。如果指定了第三个参数，

​             1、当第一个参数value是数字时， 张量的所有元素都会用该数字填充：

​              示例：

​              tensor=tf.constant(-1, shape=[2, 3])
​              sess=tf.Session()
​              with sess.as_default():
​                     print('结果是：', tensor.eval()) 

​             结果是： [[-1 -1 -1][-1 -1 -1]]

​             可以看到，输出结果是一个二维张量，第一维大小为2，第二维大小为3，全部用数字-1填充。

​             2、当第一个参数value是一个列表时，注意列表的长度必须小于等于第三个参数shape的大小（即各维

​                  大小的乘积），否则会报错：

​              示例：

​              tensor=tf.constant([1, 2, 3, 4, 5, 6, 7], shape=[2, 3])

​              Traceback (most recent call last):
​               File "<pyshell#68>", line 1, in <module>
​                tensor=tf.constant([1, 2, 3, 4, 5, 6, 7], shape=[2, 3])

​              ValueError: Too many elements provided. Needed at most 6, but received 7

​              这是因为函数会生成一个shape大小的张量，然后用value这个列表中的值一一填充shape中的元素。这

​              里列表大小为7，而shape大小为2*3=6，无法正确填充，所以发生了错误。

​               

​              而如果列表大小小于shape大小，则会用列表的最后一项元素填充剩余的张量元素：

​              示例：

​             tensor=tf.constant([1, 2], shape=[1, 4, 3])
​             sess=tf.Session()
​             with sess.as_default():
​                     print('结果是：', tensor.eval()) 

​             结果是： [[[1 2 2][2 2 2]2 2 2][2 2 2]]] 

**name:** 为这个常量起名，主要是字符串就行

​             不输入内容时：

​            tensor=tf.constant([1, 2])
​            print(tensor) 

​            Tensor("Const_16:0", shape=(2,), dtype=int32)

​            输入name时:

​           tensor=tf.constant([1, 2], name="jiayu")
​           print(tensor) 

​           Tensor("jiayu_1:0", shape=(2,), dtype=int32)

**verify_shape:**  默认为False，如果修改为True的话表示检查value的形状与shape是否相符，如果不符会报错。

​                  示例：

​                  tensor=tf.constant([[1, 2, 3], [4, 5, 6]], shape=[2, 3], verify_shape=True)

​                  以上代码value与shape都是两行三列，检查结果正确。而下面的代码会报错：

​                  tensor=tf.constant([1, 2], shape=[3, 2], verify_shape=True)

##### <5>tf.placeholder

tf.placeholder(dtype, shape=None,name=None)

placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。

**参数：**

**dtype：**数据类型，常用的是tf.float32,tf.float64等数值类型

**shape：**数据形状。默认是None，就是一维值，也可以是多维（比如[2,3], [None, 3]表示列是3，行不定）

**name：**操作名称

##### <6>tf.zeros

tf.zeros( shape, dtype=tf.float32, name=None)

该操作返回一个带有形状shape的类型为dtype张量，并且所有元素都设为零。

**参数：**

**shape：**整数、整数元组或类型为int32的1维Tensor的列表。

**dtype：**结果Tensor中元素的类型。

**name：**操作的名称

##### <7>tf.argmax

tf.argmax(input,axis)

根据axis取值的不同返回每行或者每列最大值的索引。

axis=0时比较每一列的元素，将每一列最大元素所在的索引记录下来，最后输出每一列最大元素所在的索引数组。

axis=1的时候，将每一行最大元素所在的索引记录下来，最后返回每一行最大元素所在的索引数组。

```python
test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
np.argmax(test, 0)　　　＃输出：array([3, 3, 1]
np.argmax(test, 1)　　　＃输出：array([2, 2, 0, 0]
```

##### <8>tf.stack

tf.stack( values, axis=0, name='stack')

**参数：**

- values：具有相同形状和类型的 Tensor 对象列表。
- axis：一个 int，要一起堆叠的轴，默认为第一维，负值环绕，所以有效范围是[-(R+1), R+1)。
- name：此操作的名称（可选）。

**返回值：** output与values具有相同的类型的堆叠的tensor

**可能的异常：**ValueError：如果 axis 超出范围 [ - （R + 1），R + 1），则引发此异常。



将秩为 R 的张量列表堆叠成一个秩为 (R+1) 的张量。 

将 values 中的张量列表打包成一个张量，该张量比 values 中的每个张量都高一个秩，通过沿 axis 维度打包。给定一个形状为(A, B, C)的张量的长度 N 的列表；

如果 axis == 0，那么 output 张量将具有形状(N, A, B, C)。如果 axis == 1，那么 output 张量将具有形状(A, N, B, C)。

例如：

```python
x = tf.constant([1, 4])
y = tf.constant([2, 5])
z = tf.constant([3, 6])
tf.stack([x, y, z])  # [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)
tf.stack([x, y, z], axis=1)  # [[1, 2, 3], [4, 5, 6]]
```

此函数相当于：

tf.stack([x, y, z]) = np.stack([x, y, z])

#### 2、数据类型转换

##### <1>tf.string_to_number

tf.string_to_number(string_tensor, out_type=None, name=None)

字符串转为数字

##### <2>tf.to_double

tf.to_double(x, name=’ToDouble’)

转为64位浮点类型–float64

##### <3>tf.to_float

tf.to_float(x, name=’ToFloat’)

转为32位浮点类型–float32

##### <4>tf.to_int32

tf.to_int32(x, name=’ToInt32’)

转为32位整型–int32

##### <5>tf.to_int64

tf.to_int64(x, name=’ToInt64’)

转为64位整型–int64

##### <6>tf.cast

tf.cast(x, dtype, name=None)

tf.cast()函数的作用是执行 tensorflow 中张量数据类型转换，比如读入的图片如果是int8类型的，一般在要在训练前把图像的数据格式转换为float32。

**参数：**

​         **x：**待转换的数据（张量）

​        **dtype：**目标数据类型

​        **name：**可选参数，定义操作的名称

**示例：**

​       import tensorflow as tf

​       t1 = tf.Variable([1,2,3,4,5])
​       t2 = tf.cast(t1,dtype=tf.float32)

​       print 't1: {}'.format(t1)
​       print 't2: {}'.format(t2)

​     with tf.Session() as sess:
​             sess.run(tf.global_variables_initializer())
​             sess.run(t2)
​             print t2.eval()

​            #print(sess.run(t2))

​     输出：

​         t1: <tf.Variable 'Variable:0' shape=(5,) dtype=int32_ref>

​         t2: Tensor("Cast:0", shape=(5,), dtype=float32)

​         [ 1.  2.  3.  4.  5.]

#### 3、形状操作

##### <1>tf.shape

tf.shape(input, name=None)

返回数据的shape

\# ‘t’ is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]

shape(t) ==> [2, 2, 3]

##### <2>tf.size

返回数据的元素数量

\# ‘t’ is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]

size(t) ==> 12

##### <3>tf.rank

tf.rank(input, name=None)

返回tensor的rank

注意：此rank不同于矩阵的rank，

tensor的rank表示一个tensor需要的索引数目来唯一表示任何一个元素,

也就是通常所说的 “order”,“degree”或”ndims”

\#’t’ is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]

##### <4>tf.reshape

tf.reshape(tensor, shape, name=None)

改变tensor的形状

\# tensor ‘t’ is [1, 2, 3, 4, 5, 6, 7, 8, 9]

\# tensor ‘t’ has shape [9]

reshape(t, [3, 3]) ==> 

[[1, 2, 3],

[4, 5, 6],

[7, 8, 9]]

\#如果shape有元素[-1],表示在该维度打平至一维

\# -1 将自动推导得为 9:

reshape(t, [2, -1]) ==> 

[[1, 1, 1, 2, 2, 2, 3, 3, 3],

[4, 4, 4, 5, 5, 5, 6, 6, 6]]

##### <5>tf.expand_dims

tf.expand_dims(input, dim, name=None)

插入维度1进入一个tensor中

\#该操作要求-1-input.dims()

\# ‘t’ is a tensor of shape [2]

shape(expand_dims(t, 0)) ==> [1, 2]

shape(expand_dims(t, 1)) ==> [2, 1]

shape(expand_dims(t, -1)) ==> [2, 1] <=dim <= input.dims()

#### 4、切片与合并

##### <1>tf.slice

tf.slice(input_, begin, size, name=None)

对tensor进行切片操作

其中size[i] = input.dim_size(i) - begin[i]

该操作要求 0 <= begin[i] <= begin[i] + size[i] <= Di for i in [0, n]

\#’input’ is 

\#[[[1, 1, 1], [2, 2, 2]],[[3, 3, 3], [4, 4, 4]],[[5, 5, 5], [6, 6, 6]]]

tf.slice(input, [1, 0, 0], [1, 1, 3]) ==> [[[3, 3, 3]]]

tf.slice(input, [1, 0, 0], [1, 2, 3]) ==> 

[[[3, 3, 3],

[4, 4, 4]]]

tf.slice(input, [1, 0, 0], [2, 1, 3]) ==> 

[[[3, 3, 3]],

[[5, 5, 5]]]

##### <2>tf.split

tf.split(split_dim, num_split, value,name=’split’)

沿着某一维度将tensor分离为num_split tensors

\# ‘value’ is a tensor with shape [5, 30]

\# Split ‘value’ into 3 tensors along dimension 1

split0, split1, split2 = tf.split(1, 3, value)

tf.shape(split0) ==> [5, 10]

##### <3>tf.concat

tf.concat(concat_dim, values, name=’concat’)

沿着某一维度连结tensor

t1 = [[1, 2, 3], [4, 5, 6]]

t2 = [[7, 8, 9], [10, 11, 12]]

tf.concat(0, [t1, t2]) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

tf.concat(1, [t1, t2]) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

如果想沿着tensor一新轴连结打包,那么可以：

tf.concat(axis, [tf.expand_dims(t, axis) for t in tensors])

等同于tf.pack(tensors, axis=axis)

##### <4>tf.pack

tf.pack(values, axis=0, name=’pack’)

将一系列rank-R的tensor打包为一个rank-(R+1)的tensor

\# ‘x’ is [1, 4], ‘y’ is [2, 5], ‘z’ is [3, 6]

pack([x, y, z]) => [[1, 4], [2, 5], [3, 6]] 

\# 沿着第一维pack

pack([x, y, z], axis=1) => [[1, 2, 3], [4, 5, 6]]

等价于tf.pack([x, y, z]) = np.asarray([x, y, z])

##### <5>tf.reverse

tf.reverse(tensor, dims, name=None)

沿着某维度进行序列反转

其中dim为列表，元素为bool型，size等于rank(tensor)

\# tensor ‘t’ is 

[[[[ 0, 1, 2, 3],

\#[ 4, 5, 6, 7], 

\#[ 8, 9, 10, 11]],

\#[[12, 13, 14, 15],

\#[16, 17, 18, 19],

\#[20, 21, 22, 23]]]]

\# tensor ‘t’ shape is [1, 2, 3, 4]

\# ‘dims’ is [False, False, False, True]

reverse(t, dims) ==>

[[[[ 3, 2, 1, 0],

[ 7, 6, 5, 4],

[ 11, 10, 9, 8]],

[[15, 14, 13, 12],

[19, 18, 17, 16],

[23, 22, 21, 20]]]]

##### <6>tf.transpose

tf.transpose(a, perm=None, name=’transpose’)

调换tensor的维度顺序

按照列表perm的维度排列调换tensor顺序，

如为定义，则perm为(n-1…0)

\# ‘x’ is [[1 2 3],[4 5 6]]

tf.transpose(x) ==> [[1 4], [2 5],[3 6]]

\# Equivalently

tf.transpose(x, perm=[1, 0]) ==> [[1 4],[2 5], [3 6]]

##### <7>tf.gather

tf.gather(params, indices,validate_indices=None, name=None)

合并索引indices所指示params中的切片

![](/tf.gather.png)

##### <8>tf.one_hot

tf.one_hot(indices, depth, on_value=None, off_value=None, axis=None, dtype=None, name=None)

indices = [0, 2, -1, 1]

depth = 3

on_value = 5.0 

off_value = 0.0 

axis = -1 

\#Then output is [4 x 3]: 

output = 

[5.0 0.0 0.0] // one_hot(0) 

[0.0 0.0 5.0] // one_hot(2) 

[0.0 0.0 0.0] // one_hot(-1) 

[0.0 5.0 0.0] // one_hot(1)

### （3）矩阵操作

##### <1>tf.diag

tf.diag(diagonal, name=None)

返回一个给定对角值的对角tensor

\# ‘diagonal’ is [1, 2, 3, 4]

tf.diag(diagonal) ==> 

[[1, 0, 0, 0]

[0, 2, 0, 0]

[0, 0, 3, 0]

[0, 0, 0, 4]]

##### <2>tf.diag_part

tf.diag_part(input, name=None)

功能与上面相反

##### <3>tf.trace

tf.trace(x, name=None)

求一个2维tensor足迹，即对角值diagonal之和

##### <4>tf.transpose

tf.transpose(a, perm=None, name=’transpose’)

调换tensor的维度顺序

按照列表perm的维度排列调换tensor顺序，

如为定义，则perm为(n-1…0)

\# ‘x’ is [[1 2 3],[4 5 6]]

tf.transpose(x) ==> [[1 4], [2 5],[3 6]]

\# Equivalently

tf.transpose(x, perm=[1, 0]) ==> [[1 4],[2 5], [3 6]]

##### <5>tf.matmul

tf.matmul(a, b, transpose_a=False, transpose_b=False, a_is_sparse=False, b_is_sparse=False, name=None)

矩阵相乘

##### <6>tf.matrix_determinant

tf.matrix_determinant(input, name=None)

返回方阵的行列式

##### <7>tf.matrix_inverse

tf.matrix_inverse(input, adjoint=None,name=None)

求方阵的逆矩阵，adjoint为True时，计算输入共轭矩阵的逆矩阵

##### <8>tf.cholesky

tf.cholesky(input, name=None)

对输入方阵cholesky分解，即把一个对称正定的矩阵表示成一个下三角矩阵L和其转置的乘积的分解A=LL^T

##### <9>tf.matrix_solve

tf.matrix_solve(matrix, rhs, adjoint=None,name=None)

求解tf.matrix_solve(matrix, rhs, adjoint=None, name=None)

matrix为方阵shape为[M,M],rhs的shape为[M,K]，output为[M,K]

### （4）复数操作

##### <1>tf.complex

tf.complex(real, imag, name=None)

将两实数转换为复数形式

\# tensor ‘real’ is [2.25, 3.25]

\# tensor imag is [4.75, 5.75]

tf.complex(real, imag) ==> [[2.25 + 4.75j], [3.25 + 5.75j]]

##### <2>tf.complex_abs

tf.complex_abs(x, name=None)

计算复数的绝对值，即长度。

\# tensor ‘x’ is [[-2.25 + 4.75j], [-3.25 + 5.75j]]

tf.complex_abs(x) ==> [5.25594902, 6.60492229]

##### <3>tf.conj

tf.conj(input, name=None)

计算共轭复数

##### <4>tf.imag\tf.real

提取复数的虚部和实部

##### <5>tf.fft

计算一维的离散傅里叶变换，输入数据类型为complex64

### （5）归约计算

##### <1>tf.reduce_sum

tf.reduce_sum(input_tensor, reduction_indices=None, keep_dims=False, name=None)

计算输入tensor元素的和，或者安照reduction_indices指定的轴进行求和

\# ‘x’ is [[1, 1, 1]

\# [1, 1, 1]]

tf.reduce_sum(x) ==> 6

tf.reduce_sum(x, 0) ==> [2, 2, 2]

tf.reduce_sum(x, 1) ==> [3, 3]

tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]

tf.reduce_sum(x, [0, 1]) ==> 6

##### <2>tf.reduce_prod

tf.reduce_prod(input_tensor, reduction_indices=None, keep_dims=False, name=None)

计算输入tensor元素的乘积，或者安照reduction_indices指定的轴进行求乘积

##### <3>tf.reduce_min

tf.reduce_min(input_tensor, reduction_indices=None, keep_dims=False, name=None)

求tensor中最小值

##### <4>tf.reduce_max

tf.reduce_max(input_tensor, reduction_indices=None, keep_dims=False, name=None)

求tensor中最大值

##### <5>tf.reduce_mean

tf.reduce_mean(input_tensor,axis=None , keep_dims=False, name=None,reduction_indices=None)

tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。

**参数：**

**input_tensor：**输入的tensor

**axis：**指定的轴，如果不指定，则计算所有元素的平均值

**keep_dims：**是否降维度，设置为Ture，输出的结果保持输入tensor的形状，设置为False，输出结果会降低维度

**name：**操作的名称

**reduction_indices：**在以前的版本中用来指定轴，现在已经弃用

##### <6>tf.reduce_all

tf.reduce_all(input_tensor, reduction_indices=None, keep_dims=False, name=None)

对tensor中各个元素求逻辑’与’

\# ‘x’ is 

\# [[True, True]

\# [False, False]]

tf.reduce_all(x) ==> False

tf.reduce_all(x, 0) ==> [False, False]

tf.reduce_all(x, 1) ==> [True, False]

##### <7>tf.reduce_any

tf.reduce_any(input_tensor, reduction_indices=None, keep_dims=False, name=None)

对tensor中各个元素求逻辑’或’

##### <8>tf.accumulate_n

tf.accumulate_n(inputs, shape=None, tensor_dtype=None, name=None)

计算一系列tensor的和

\# tensor ‘a’ is [[1, 2], [3, 4]]

\# tensor b is [[5, 0], [0, 6]]

tf.accumulate_n([a, b, a]) ==> [[7, 4], [6, 14]]

##### <9>tf.cumsum

tf.cumsum(x, axis=0, exclusive=False, reverse=False, name=None)

求累积和

tf.cumsum([a, b, c]) ==> [a, a + b, a + b + c]

tf.cumsum([a, b, c], exclusive=True) ==> [0, a, a + b]

tf.cumsum([a, b, c], reverse=True) ==> [a + b + c, b + c, c]

tf.cumsum([a, b, c], exclusive=True, reverse=True) ==> [b + c, c, 0]

### （6）分割操作

##### <1>tf.segment_sum

tf.segment_sum(data, segment_ids, name=None)

根据segment_ids的分段计算各个片段的和

其中segment_ids为一个size与data第一维相同的tensor

其中id为int型数据，最大id不大于size

c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])

tf.segment_sum(c, tf.constant([0, 0, 1]))

==>[[0 0 0 0] 

[5 6 7 8]]

上面例子分为[0,1]两id,对相同id的data相应数据进行求和,

并放入结果的相应id中，

且segment_ids只升不降

##### <2>tf.segment_prod

tf.segment_prod(data, segment_ids, name=None)

根据segment_ids的分段计算各个片段的积

##### <3>tf.segment_min

tf.segment_min(data, segment_ids, name=None)

根据segment_ids的分段计算各个片段的最小值

##### <4>tf.segment_max

tf.segment_max(data, segment_ids, name=None)

根据segment_ids的分段计算各个片段的最大值

##### <5>tf.segment_mean

tf.segment_mean(data, segment_ids, name=None)

根据segment_ids的分段计算各个片段的平均值

##### <6>tf.unsorted_segment_sum

tf.unsorted_segment_sum(data, segment_ids,num_segments, name=None)

与tf.segment_sum函数类似，不同在于segment_ids中id顺序可以是无序的

##### <7>tf.sparse_segment_sum

tf.sparse_segment_sum(data, indices, segment_ids, name=None)

输入进行稀疏分割求和

c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])

\# Select two rows, one segment.

tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 0])) 

==> [[0 0 0 0]]

对原data的indices为[0,1]位置的进行分割，

并按照segment_ids的分组进行求和

### （7）序列比较与索引提取

##### <1>tf.argmin

tf.argmin(input, dimension, name=None)

返回input最小值的索引index

##### <2>tf.argmax

tf.argmax(input, dimension, name=None)

返回input最大值的索引index

##### <3>tf.listdiff

tf.listdiff(x, y, name=None)

返回x，y中不同值的索引

##### <4>tf.where

tf.where(input, name=None)

返回bool型tensor中为True的位置

\# ‘input’ tensor is 

\#[[True, False]

\#[True, False]]

\# ‘input’ 有两个’True’,那么输出两个坐标值.

\# ‘input’的rank为2, 所以每个坐标为具有两个维度.

where(input) ==>

[[0, 0],

[1, 0]]

##### <5>tf.unique

tf.unique(x, name=None)

返回一个元组tuple(y,idx)，y为x的列表的唯一化数据列表，

idx为x数据对应y元素的index

\# tensor ‘x’ is [1, 1, 2, 4, 4, 4, 7, 8, 8]

y, idx = unique(x)

y ==> [1, 2, 4, 7, 8]

idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]

##### <6>tf.invert_permutation

tf.invert_permutation(x, name=None)

置换x数据与索引的关系

\# tensor x is [3, 4, 0, 2, 1]

invert_permutation(x) ==> [2, 4, 3, 0, 1]

### （8）神经网络

#### 1、卷积函数

##### <1>tf.nn.convolution

tf.nn.convolution(input, filter, padding,strides=None, dilation_rate=None, name=None, data_format =None)

这个函数计算 N 维卷积的和

##### <2>tf.nn.conv2d

tf.nn.conv2d(input, filter, strides, padding,use_cudnn_on_gpu=None, data_format=None, name=None)

这个函数的作用是对一个四维的输入数据input 和四维的卷积核 filter 进行操作，然 后对输入数据进行一个二维的卷积操作，最后得到卷积之后的结果。

**参数：**

**input**：指需要做卷积的输入图像，要求是一个Tensor,具有[batch, in_height, in_width, in_channels]这样的shape，具体含义是[训练时一个batch的图片数量，图片高度，图片宽度，图像通道数]，注意这是一个4维的Tensor，要求类型为float32和float64其中之一

**filter** : 相当于CNN中的卷积核，要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同。另外，需要注意的是，第三维in_channels就是参数input的第四维。

**strides：**一个长度是4的一维整数类型数组，每一维度对应的是input中每一维的对应移动步数，比如，strides[1]对应input[1]的移动步数

**padding：**string类型的量，取值为SAME或者VALID

​                 padding='SAME':仅适用于全尺寸操作，即输入数据维度和输出数据维度相同

​                 padding='VALID':适用于部分窗口，即输入数据维度和输出数据维度不同

**use_cudnn_on_gpu**：bool类型，是否使用cudnn加速，默认为true

**name：**可选，为这个操作取一个名字

**使用示例：**

input_data = tf.Variable( np.random.rand(10,9,9,3), dtype = np.float32 )

filter_data = tf.Variable( np.random.rand(2, 2, 3, 2), dtype = np.float32)

y = tf.nn.conv2d(input_data, filter_data, strides = [1, 1, 1, 1], padding = 'SAME')

打印出 tf.shape(y)的结果是[10 9 9 2]

##### <3>tf.nn.depthwise_conv2d

tf.nn.depthwise_conv2d (input, filter, strides, padding, rate=None, name=None,data_format= None)

这个函数输入张量的数据维度是[batch, in_height,in_width, in_channels]，卷积核的维度是 [filter_height, filter_width, in_channels, channel_multiplier]，在通道 in_channels 上面的卷积深度是 1，depthwise_conv2d 函数将不同的卷积核独立地应用在 in_channels 的每个通道上（从通道 1 到通道 channel_multiplier），然后把所以的结果进行汇总。最后输出通道的总数是 in_channels * channel_multiplier。

**使用示例：**

input_data = tf.Variable( np.random.rand(10, 9, 9, 3), dtype = np.float32 ) 

filter_data = tf.Variable( np.random.rand(2, 2, 3, 5), dtype = np.float32)

y = tf.nn.depthwise_conv2d(input_data,filter_data,strides= [1, 1, 1, 1], padding = 'SAME')

这里打印出 tf.shape(y)的结果是[10 9 9 15]。

##### <4>tf.nn.separable_conv2d

tf.nn.separable_conv2d (input, depthwise_filter, pointwise_filter, strides, padding, rate=None, name=None, data_format=None)

此函数是利用几个分离的卷积核去做卷积。在这个API 中，将应用一个 二维的卷积核，在每个通道上，以深度 channel_multiplier 进行卷积。

特殊参数：

**depthwise_filter：** 一个张量。数据维度是四维[filter_height, filter_width, in_channels,   channel_multiplier]。

​                                     其中，in_channels的卷积深度是1。

**pointwise_filter：**一个张量。数据维度是四维[1, 1, channel_multiplier * in_channels,out_channels]。

​                                   其中，pointwise_filter 是在 depthwise_filter 卷积之后的混合卷积。

**使用示例：**

 input_data = tf.Variable( np.random.rand(10,9, 9, 3), dtype = np.float32 )

depthwise_filter = tf.Variable( np.random.rand(2, 2, 3, 5), dtype = np.float32) 

pointwise_filter = tf.Variable( np.random.rand(1, 1, 15, 20), dtype = np.float32)

\# out_channels >= channel_multiplier * in_channels

y = tf.nn.separable_conv2d(input_data, depthwise_filter, pointwise_filter,strides = [1, 1, 1, 1], padding = 'SAME')

这里打印出 tf.shape(y)的结果是[10 9 9 20]。

##### <5>tf.nn.atrous_conv2d

tf.nn.atrous_conv2d(value, filters,rate, padding, name=None)

计算 Atrous 卷积，又称孔卷 积或者扩张卷积。

**使用示例：**

input_data = tf.Variable( np.random.rand(1,5,5,1), dtype = np.float32 ) 

filters = tf.Variable( np.random.rand(3,3,1,1), dtype = np.float32)

y = tf.nn.atrous_conv2d(input_data, filters, 2, padding='SAME')

这里打印出tf.shape(y)的结果是[1 5 5 1]。

##### <6>tf.nn.conv2d_transpose

tf.nn.conv2d_transpose(value,filter,output_shape,strides,padding='SAME',data_format='NHWC',name=None)

在解卷积网络（deconvolutional network）中有时称为“反卷积”，但实际上是 conv2d 的转置，而不是实际的反卷积。

**特殊参数：**

**output_shape：**一维的张量，表示反卷积运算后输出的形状

输出：和 value 一样维度的 Tensor

**使用示例：**

x = tf.random_normal(shape=[1,3,3,1]) 

kernel = tf.random_normal(shape=[2,2,3,1])

y = tf.nn.conv2d_transpose(x,kernel,output_shape=[1,5,5,3],strides=[1,2,2,1],padding="SAME")

这里打印出 tf.shape(y)的结果是[1 5 5 3]。

##### <7>tf.nn.conv1d

tf.nn.conv1d(value, filters, stride, padding,use_cudnn_on_gpu=None, data_format=None, name=None)

和二维卷积类似。这个函数是用来计算给定三维的输入和过滤器的情况下的一维卷 积。不同的是，它的输入是三维，如[batch, in_width, in_channels]。卷积核的维度也是三维，少 了一维 filter_height，如 [filter_width, in_channels, out_channels]。stride 是一个正整数，代表卷积 核向右移动每一步的长度。

##### <8>tf.nn.conv3d

tf.nn.conv3d(input, filter, strides, padding, name=None)

和二维卷积类似。这个函数用来计 算给定五维的输入和过滤器的情况下的三维卷积。和二维卷积相对比：

●      input 的 shape 中多了一维 in_depth，形状为 Shape[batch, in_depth, in_height, in_width, in_channels]；

●      filter 的 shape 中多了一维 filter_depth，由 filter_depth, filter_height, filter_width 构成了卷 积核的大小；

●      strides 中多 了一维， 变 为 [strides_batch, strides_depth, strides_height, strides_width, strides_channel]，必须保证 strides[0] = strides[4] = 1。

##### <9>tf.nncon3d_transpose

tf.nn.conv3d_transpose(value, filter,output_shape, strides, padding='SAME',  name=None)

和二维反卷积类似

#### 2、池化函数

##### <1>tf.nn.avg_pool

tf.nn.avg_pool(value, ksize, strides, padding,data_format='NHWC', name=None)

这个函 数计算池化区域中元素的平均值。

**参数：**

**value：**一个四维的张量。数据维度是[batch, height, width, channels]

**ksize：**一个长度不小于 4 的整型数组。每一位上的值对应于输入数据张量中每一维的窗口对应值

**strides：**一个长度不小于 4 的整型数组。该参数指定滑动窗口在输入数据张量每一维上的步长

**padding：**一个字符串，取值为 SAME 或者 VALID

**data_format:** 'NHWC'代表输入张量维度的顺序，N 为个数，H 为高度，W 为宽度，C 为通道数（RGB 三

通道或者灰度单通道）

**name**（可选）：为这个操作取一个名字

**使用示例：**

input_data = tf.Variable( np.random.rand(10,6,6,3), dtype = np.float32 ) 

filter_data = tf.Variable( np.random.rand(2, 2, 3, 10), dtype = np.float32)

y = tf.nn.conv2d(input_data, filter_data, strides = [1, 1, 1, 1], padding = 'SAME')

output = tf.nn.avg_pool(value = y, ksize = [1, 2, 2, 1], strides = [1, 1, 1, 1],padding ='SAME')

上述代码打印出 tf.shape(output)的结果是[10 6 6 10]。计算输出维度的方法是：shape(output)
=(shape(value) - ksize + 1) / strides。

##### <2>tf.nn.max_pool

tf.nn.max_pool(value, ksize, strides, padding,data_format='NHWC', name=None)

这个函数是计算池化区域中元素的最大值。

**参数：**

**value：**需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map,依然是[batch,height,width,channels]这样的shape

**ksize：**池化窗口的大小，取一个4维向量，一般是[1, height, width, 1],因为不想在batch和channels上做池化，所以这两个维度设为1

**strides：**和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride, stride, 1]

**padding：**和卷积类似，可以取‘VALID’和‘SAME’

**使用示例：**

input_data = tf.Variable( np.random.rand(10,6,6,3), dtype = np.float32 ) 

filter_data = tf.Variable( np.random.rand(2, 2, 3, 10), dtype = np.float32)

y = tf.nn.conv2d(input_data, filter_data, strides = [1, 1, 1, 1], padding = 'SAME')

output = tf.nn.max_pool(value = y, ksize = [1, 2, 2, 1], strides = [1, 1, 1, 1],padding ='SAME')

上述代码打印出 tf.shape(output)的结果是[10 6 6 10]。

##### <3>tf.nn.max_pool_with_argmax

tf.nn.max_pool_with_argmax(input, ksize,strides, padding, Targmax = None, name=None)

这个函数的作用是计算池化区域中元素的最大值和该最大值所在的位置。

在计算位置 argmax 的时候，我们将 input 铺平了进行计算，所以，如果 input = [b, y, x, c]， 那么索引位置是(( b * height + y) * width +x) * channels + c。

**使用示例：**该函数只能在 GPU 下运行，在 CPU 下没有对应的函数实现

input_data = tf.Variable( np.random.rand(10,6,6,3), dtype = tf.float32 ) 

filter_data = tf.Variable( np.random.rand(2, 2, 3, 10), dtype = np.float32)

y = tf.nn.conv2d(input_data, filter_data, strides = [1, 1, 1, 1], padding = 'SAME')

output, argmax = tf.nn.max_pool_with_argmax(input = y, ksize = [1, 2, 2, 1],strides = [1, 1, 1, 1], padding = 'SAME')

返回结果是一个张量组成的元组（output, argmax），output 表示池化区域的最大值；argmax 的数据类型是 Targmax，维度是四维。

##### <4>tf.nn.avg_pool3d

三维下的平均池化

##### <5>tf.nn.max_pool3d

三维下的最大池化

##### <6>tf.nn.fractional_avg_pool

三维下的平均池化

##### <7>tf.nn.fractional_max_pool

三维下的最大池化

##### <8>tf.nn.pool

tf.nn.pool(input, window_shape, pooling_type,padding, dilation_rate=None, strides=None, name=None, data_format=None)

这个函数执行一个 N 维的池化操作。

#### 3、激活函数

激活函数（activation function）运行时激活神经网络中某一部分神经元，将激活信息向后传入下一层的神经网络。神经网络之所以能解决非线性问题（如语音、图像识别），本质上就 是激活函数加入了非线性因素，弥补了线性模型的表达力，把“激活的神经元的特征”通过函数保留并映射到下一层。

**注：**

1、软饱和是指激活函数 *h*(*x*)在取值趋于无穷大时，它的一阶导数趋于 0。硬饱和是指当|*x*| > *c* 时，其中 *c* 为常数，*f* *'*(*x*)=0。relu 就是一类左侧硬饱和激活函数。

2、梯度消失是指在更新模型参数时采用链式求导法则反向求导，越往前梯度越小。最终的结果是到达一定深度后梯 度对模型的更新就没有任何贡献了。

3、激活函数的选择：当输入数据特征相差明显时，用 tanh 的效果会很好，且在循环过程中会不断扩大特征效果并显示出来。当特征相差不明显时，sigmoid 效果比较好。同时，用 sigmoid 和 tanh 作为激活函数时， 需要对输入进行规范化，否则激活后的值全部都进入平坦区，隐层的输出会全部趋同，丧失原有 的特征表达。而 relu 会好很多，有时可以不需要输入规范化来避免上述情况。 因此，现在大部分的卷积神经网络都采用 relu 作为激活函数。我估计大概有 85%～90%的神经网 络会采用 ReLU，10%～15%的神经网络会采用 tanh，尤其用在自然语言处理上。

##### <1>tf.nn.relu

tf.nn.relu(features,name=None)

这个函数的作用是计算激活函数relu，即f(x)=max (x,0),即将矩阵中每行的非最大值置0

![](/relu and softplus.png)

如图，relu 在 *x*<0 时硬饱和。由于 *x*>0 时 导数为 1，所以，relu能够在 *x*>0时保持梯度不衰减，从 而缓解梯度消失问题，还能够更很地收敛，并提供了神经网络的稀疏表达能力。但是，随着训练的进行， 部分输入会落到硬饱和区，导致对应的权重无法更新， 称为“神经元死亡”。

使用示例：

a = tf.constant([-1.0, 2.0]) 

with tf.Session() as sess:

​         b = tf.nn.relu(a) 

​         print sess.run(b)

##### <2>tf.nn.softplus

softplus可以看做是ReLU的平滑版本。定义为f(x)=log(1+exp(x)).

##### <3>tf.nn.relu6

tf.nn.relu6(features, name=None)

被定义为min(max(features, 0), 6)

##### <4>tf.nn.crelu

tf.nn.crelu(features, name=None)

##### <5>tf.nn.sigmoid

  sigmoid 函数的优点在于，它的输出映射在(0,1)内，单调连续，非常适合用作输出层，并且 求导比较容易。但是，它也有缺点，因为软饱和性，一旦输入落入饱和区，*f'*(*x*)就会变得接近 于 0，很容易产生梯度消失。

![](/sigmoid.png)


​                                                                          ![](/sigmoid函数式.png)

**使用示例：**

a = tf.constant([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])

sess = tf.Session()

print sess.run(tf.sigmoid(a))

##### <6>tf.nn.tanh

![](/tanh.png)

![](/tanh函数式.png)

tanh 函数也具有软饱和性。因为它的输出以 0 为中心，收敛速度比 sigmoid 要很。但是仍 无法解决梯度消失的问题。

##### <7>tf.nn.elu

##### <8>tf.nn.bias_add

##### <9>tf.nn.softsign

##### <10>tf.nn.dropout

tf.nn.dropout(x, keep_prob, noise_shape, seed=None, name=None)

**参数：**

**x：**输入Tensor

**keep_prob：**float类型，每个元素被保留下来的概率

​                      一个神经元将以概率 keep_prob 决定是否被抑制。如果被抑制，该神经元的输出就为0；如果不被抑制，那么该神经元的输出值将被放大到原来的1/keep_prob 倍。

**noise_shape：**一个1维的int32张量，代表了随机产生“保留/丢弃”标志的shape

​                        在默认情况下，每个神经元是否被抑制是相互独立的。但是否被抑制也可以通过noise_shape 来 调节。当 noise_shape[i] == shape(x)[i]时，x 中的元素是相互独立的。如果 shape(x) =[k, l, m, n]， x 中的维度的顺序分别为批、行、列和通道，如果 noise_shape = [k, 1, 1, n]，那么每个批和通道 都是相互独立的，但是每行和每列的数据都是关联的，也就是说，要不都为 0，要不都还是原 来的值。 

**seed：**整形变量，随机数种子

**使用示例：**

a = tf.constant([[-1.0, 2.0, 3.0, 4.0]]) 

with tf.Session() as sess:

b = tf.nn.dropout(a, 0.5, noise_shape = [1,4])

print sess.run(b)

b = tf.nn.dropout(a, 0.5, noise_shape = [1,1]) 

print sess.run(b)

**注：**dropout 在论文中最我被提出时是这么做的：在训练的时候用概率p 丢弃，然后在预测的时候，所有参数按比例缩 小，也就是乘以 *p*。

在各种深度学习框架（如Keras、TensorFlow）的实现中，都是用反向 ropout 来代替 dropout。 也就是这里所说的，在训练的时候一边 dropout，然后再按比例放大，也就是乘以 1/*p*，然后在预测的时候，不做 任务处理。

#### 4、分类函数

##### <1>tf.nn.sigmoid_cross_entropy_with_logits

tf.nn.sigmoid_cross_entropy_with_logits(logits,targets, name=None)：

**输入**：logits:[batch_size, num_classes],targets:[batch_size, size].

logits 用最后一层的输入即可

最后一层不需要进行 sigmoid 运算，此函数内部进行了 sigmoid 操作

**输出**：loss [batch_size, num_classes]

**这个函数的输入要格外注意，如果采用此函数作为损失函数，在神经网络的最后一层不需 要进行 sigmoid 运算。**

##### <2>tf.nn.softmax

tf.nn.softmax(logits,dim=-1, name=None)

计算 Softmax 激活，也就是 softmax = exp(logits) / reduce_sum(exp(logits), dim)。

##### <3>tf.nn.log_softmax

tf.nn.log_softmax(logits,dim=-1, name=None)

计算 log softmax 激活，也就是 logsoftmax = logits -log(reduce_sum(exp(logits), dim))

##### <4>tf.nn.softmax_cross_entropy_with_logits

tf.nn.softmax_cross_entropy_with_logits(_sentinel=None,labels=None, logits=None, dim=-1, name =None)

**输入**：logits and labels 均为[batch_size, num_classes]

**输出**： loss:[batch_size], 里面保存的是 batch 中每个样本的交叉熵

##### <5>tf.nn.sparse_softmax_cross_entropy_with_logits

tf.nn.sparse_softmax_cross_entropy_with_logits(logits,labels, name=None)

logits 是神经网络最后一层的结果

**输入**：logits: [batch_size, num_classes] labels: [batch_size]，必须在[0, num_classes]

**输出**：loss [batch_size]，里面保存是 batch 中每个样本的交叉熵

#### 5、数据标准化

##### <1>tf.nn.l2_normalize

tf.nn.l2_normalize(x, dim, epsilon=1e-12,name=None)

对维度dim进行L2范式标准化

**输出**：output = x / sqrt(max(sum(x**2), epsilon))

##### <2>tf.nn.sufficient_statistics

tf.nn.sufficient_statistics(x, axes, shift=None, keep_dims=False, name=None)

计算与均值和方差有关的完全统计量

**输出**：返回4维元组,（元素个数，元素总和，*元素的平方和，*shift结果）

##### <3>tf.nn.normalize_moments

tf.nn.normalize_moments(counts, mean_ss,variance_ss, shift, name=None)

基于完全统计量计算均值和方差

##### <4>tf.nn.moments

tf.nn.moments(x, axes, shift=None, name=None, keep_dims=False)

直接计算均值与方差

#### 6、损失/代价函数

###### （1）二次代价函数

![](/二次代价函数.jpg)

其中，C代表代价函数，x代表样本，y表示实际值，a表示输出值，n表示样本的总数。

当只有一个样本时，此时的二次代价函数为：$C=\frac{(y-a)^2}{2}​$,其中$a=\sigma(Z)​$,$Z=\sum{W_j*X_j+b}​$

假如我们使用梯度下降法（Gradient descent）来调整权值参数的大小，权值w和偏置b的梯度推导：

![](/梯度下降公式.jpg)

其中，z表示神经元的输入，$\sigma$表示激活函数。w和b的梯度跟激活函数的梯度成正比，激活函数的梯度越大，w和b

的大小调整越快，训练收敛得就越快。

假设我们的激活函数是sigmoid函数：

![](/sigmoid函数图.png)

**y - a**

假设我们目标是收敛到1。A点为0.82离目标比较远，梯度比较大，权值调整比较大。

​                                             B点为0.98离目标比较近，梯度比较小，权值调整比较小。

​                                             调整方案合理。

假设我们目标是收敛到0。A点为0.82离目标比较近，梯度比较大，权值调整比较大。

​                                             B点为0.98离目标比较远，梯度比较小，权值调整比较小。

​                                             调整方案不合理。

###### （2）交叉熵代价函数（cross-entropy）

![](/交叉熵代价函数.jpg)

其中， C表示代价函数 ，x表示样本， y表示实际值， a表示输出值， n表示样本的总数。

$a=\sigma(Z)​$,$Z=\sum{W_j*X_j+b}​$

$\frac{d}{dx}\sigma{(Z)}=\sigma(Z)（1-\sigma(Z))​$

![](/交叉熵梯度下降.jpg)

![](/交叉熵梯度下降2.jpg)

权值和偏置值的调整与$\frac{d}{dx}\sigma{(Z)}$无关,另外，梯度公式中的$\sigma(Z)-y$表示输出值与实际值的误差。所以当误差越大时，梯度就越大，参数w和b的调整就越快，训练的速度也就越快。



如果输出神经元是线性的，那么二次代价函数是一种合适的选择。

如果输出神经元是S型函数（比如sigmoid函数），那么比较适合用交叉熵代价函数。

###### （3）对数释然代价函数（log-likelihood cost）

![](/对数释然代价函数.png)

对数释然函数常用来作为softmax回归的代价函数，如果输出层神经元是sigmoid函数，可以采用交叉熵代价函数。而深度学习中更普遍的做法是将softmax作为最后一层，此时常用的代价函数是 对数释然代价函数。



对数释然函数与softmax函数的组合和交叉熵与sigmoid函数的组合非常类似。

对数释然代价函数在二分类时可以简化为交叉熵代价函数的形式。



在Tensorflow中用：

tf.nn.sigmoid_cross_entropy_with_logits()来表示跟sigmoid搭配使用的交叉熵。

tf.nn.softmax_cross_entropy_with_logits()来表示跟softmax搭配使用的交叉熵。

##### <1>tf.nn.l2_loss

tf.nn.l2_loss(t, name=None)

**输出**：output = sum(t ** 2) / 2

#### 7、训练函数

以下主要是对tensorflow的模型训练Training与测试Testing等相关函数的讲解。

| 操作组   | 操作                                                         |
| -------- | ------------------------------------------------------------ |
| Training | Optimizers，Gradient Computation，Gradient Clipping，Distributed execution |
| Testing  | Unit tests，Utilities，Gradient checking                     |

函数training()通过梯度下降法为最小化损失函数增加了相关的优化操作，在训练过程中，先实例化一个优化函数，比如 tf.train.GradientDescentOptimizer，并基于一定的学习率进行梯度优化训练：

```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
```

然后，可以设置 一个用于记录全局训练步骤的单值。

以及使用minimize()操作，该操作不仅可以优化更新训练的模型参数，也可以为全局步骤(global step)计数。与其他tensorflow操作类似，这些训练操作都需要在tf.session会话中进行。

```python
global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)
```

##### <1>常用的optimizer优化类

![](/各种优化器对比2.gif)

![](/各种优化器对比.gif)

**1、各种优化函数**

tf中各种优化类提供了为损失函数计算梯度的方法，其中包含比较经典的优化算法，比如GradientDescent 和Adagrad。

###### **1、class tf.train.Optimizer**

| 操作                                                         | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| class tf.train.Optimizer                                     | 基本的优化类，该类不常常被直接调用，而较多使用其子类， 比如**GradientDescentOptimizer**, **AdagradOptimizer** 或者**MomentumOptimizer** |
| tf.train.Optimizer.\__init__(use_locking, name)              | 创建一个新的优化器， 该优化器必须被其子类(subclasses)的构造函数调用 |
| tf.train.Optimizer.minimize(loss, global_step=None,  var_list=None, gate_gradients=1,  aggregation_method=None, colocate_gradients_with_ops=False,  name=None, grad_loss=None) | 添加操作节点，用于最小化loss，并更新var_list 。该函数是简单的合并了compute_gradients()与apply_gradients()函数， 返回为一个优化更新后的var_list，如果global_step非None，该操作还会为global_step做自增操作 |
| tf.train.Optimizer.compute_gradients(loss,var_list=None, gate_gradients=1, aggregation_method=None,  colocate_gradients_with_ops=False, grad_loss=None) | 对var_list中的变量计算loss的梯度 ,该函数为函数minimize()的第一部分，返回一个以元组(gradient, variable)组成的列表 |
| tf.train.Optimizer.apply_gradients(grads_and_vars, global_step=None, name=None) | 将计算出的梯度应用到变量上，是函数minimize()的第二部分，返回一个应用指定的梯度的操作Operation，对global_step做自增操作 |
| tf.train.Optimizer.get_name()                                | 获取名称                                                     |

**class tf.train.Optimizer的用法示例：**

```python
# Create an optimizer with the desired parameters.
opt = GradientDescentOptimizer(learning_rate=0.1)
# Add Ops to the graph to minimize a cost by updating a list of variables.
# "cost" is a Tensor, and the list of variables contains tf.Variable objects.
opt_op = opt.minimize(cost, var_list=<list of variables>)
# Execute opt_op to do one step of training:
opt_op.run()
```

###### 2、class tf.train.GradientDescentOptimizer

这个类是实现梯度下降算法的优化器。(结合理论可以看到，这个构造函数需要的一个**学习率**就行了)

**梯度下降法不能避免鞍点**

**标准梯度下降法：**

标准梯度下降先计算所有样本汇总误差，然后根据总误差来更新权值。

**随机梯度下降法：**

随机梯度下降随机抽取一个样本来计算误差，然后更新权值。

**批量梯度下降法：**

批量梯度下降算是一种折中的方案，从总样本中选取一个批次（比如一共有10000个样本，随机选取100个样本作为一个batch），然后计算这个batch的总误差，根据总误差来更新权值。



W：要训练的参数

J(W)：代价函数

$\nabla_w$J(W)：代价函数的梯度

η：学习率

W =W−η *$\nabla_w$ J(W;$x^{(i)}$;$y^{(i)}$)



- **\__init__(learning_rate, use_locking=False,name=’GradientDescent’)**

作用：创建一个梯度下降优化器对象 
**参数：** 
**learning_rate:** A Tensor or a floating point value. 要使用的学习率 
**use_locking:** 要是True的话，就对于更新操作（update operations.）使用锁 

**name:** 名字，可选，默认是”GradientDescent”.

- **compute_gradients(loss,var_list=None,gate_gradients=GATE_OP,aggregation_method=None,colocate_gradients_with_ops=False,grad_loss=None)**

作用：对于在变量列表（var_list）中的变量计算对于损失函数的梯度,这个函数返回一个（梯度，变量）对的列表，其中梯度就是相对应变量的梯度了。这是minimize()函数的第一个部分， 
**参数：** 
**loss:** 待减小的值 
**var_list:** 默认是在GraphKey.TRAINABLE_VARIABLES. 
**gate_gradients:** How to gate the computation of gradients. Can be GATE_NONE, GATE_OP, or GATE_GRAPH. 
**aggregation_method:** Specifies the method used to combine gradient terms. Valid values are defined in the class AggregationMethod. 
**colocate_gradients_with_ops:** If True, try colocating gradients with the corresponding op. 

**grad_loss:** Optional. A Tensor holding the gradient computed for loss.

- **apply_gradients(grads_and_vars,global_step=None,name=None)**

作用：把梯度“应用”（Apply）到变量上面去。其实就是按照梯度下降的方式加到上面去。这是minimize（）函数的第二个步骤。 返回一个应用的操作。 
**参数:** 
**grads_and_vars:** compute_gradients()函数返回的(gradient, variable)对的列表 
**global_step:** Optional Variable to increment by one after the variables have been updated. 

**name:** 可选

- **minimize(loss,global_step=None,var_list=None,gate_gradients=GATE_OP,aggregation_method=None,colocate_gradients_with_ops=False,name=None,grad_loss=None)**

作用：非常常用的一个函数 
通过更新var_list来减小loss，这个函数就是前面compute_gradients() 和apply_gradients().的结合

###### 3、class tf.train.MomentumOptimizer

- **Momentum：**

![](/Momentum.png)

- **NAG( Nesterov accelerated gradient )**：

![](/NAG1.png)

![](/NAG2.png)



**f.train.MomentumOptimizer.\__init__(learning_rate, momentum, use_locking=False, name=’Momentum’, use_nesterov=False)**

learning_rate: A Tensor or a floating point value. The learning rate. 
momentum: A Tensor or a floating point value. The momentum. 
use_locking: If True use locks for update operations. 

name: Optional name prefix for the operations created when applying gradients. Defaults to “Momentum”.

###### 4、class tf.train.AdagradOptimizer

![](/Adagrad.png)

![](/Adagrad2.png)



**f.train.AdagradOptimizer.__init__(learning_rate, initial_accumulator_value=0.1, use_locking=False, name=’Adagrad’)**

learning_rate: A Tensor or a floating point value. The learning rate.
initial_accumulator_value: A floating point value. Starting value for the accumulators, must be positive.
use_locking: If True use locks for update operations.

name: Optional name prefix for the operations created when applying gradients. Defaults to "Adagrad".

###### 5、class tf.train.FtrlOptimizer

###### 6、class tf.train.ProximalGradientDescentOptimizer

###### 7、class tf.train.ProximalAdagradOptimizer

###### 8、class tf.train.RMSPropOptimizer

![](/RMSprop.png)

RMSprop借鉴了一些Adagrad的思想，不过这里RMSprop只用到了前t-1次的梯度平方的平均值加上当前梯度的平方的和的开平方作为学习率的分母。这样RMSprop不会出现学习率越来越低的问题。而且也能自己调节学习率，并且可以有一个比较好的效果。

###### 9、class tf.train.AdadeltaOptimizer

![](/Adadelta.png)

**使用Adadelta我们甚至不需要设置一个默认学习率，在Adadelta不需要使用学习率也可以达到一个非常好的效果。**

实现了 Adadelta算法的优化器，可以算是下面的Adagrad算法改进版本

**tf.train.AdadeltaOptimizer.init(learning_rate=0.001, rho=0.95, epsilon=1e-08, use_locking=False, name=’Adadelta’)**

**作用：**构造一个使用Adadelta算法的优化器 

**参数：** 
learning_rate: tensor或者浮点数，学习率 

rho: tensor或者浮点数. The decay rate. 

epsilon: A Tensor or a floating point value. A constant epsilon used to better conditioning the grad update. 

use_locking: If True use locks for update operations. 

name: 【可选】这个操作的名字，默认是”Adadelta”

###### 10、class tf.train.AdagradDAOptimizer

###### 11、class tf.train.AdamOptimizer

![](/Adam.png)

**就像Adadelta和RMSprop一样，Adam会存储之前衰减的平方梯度，同时它也会保存之前衰减的梯度。经过一些处理之后再使用类似Adadelta和RMSprop的方式更新参数。**

**tf.train.AdamOptimizer.__init__(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name=’Adam’)**

Initialization:

m_0 <- 0 (Initialize initial 1st moment vector) 

v_0 <- 0 (Initialize initial 2nd moment vector) 

t <- 0 (Initialize timestep) 

The update rule for variable with gradient g uses an optimization described at the end of section2 of the 

paper:

t <- t + 1 

lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)

m_t <- beta1 * m_{t-1} + (1 - beta1) * g __
v_t <- beta2 * v_{t-1} + (1 - beta2) * g * g 

variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon) 

The default value of 1e-8 for epsilon might not be a good default in general. For example, when training an Inception network on ImageNet a current good choice is 1.0 or 0.1.

Note that in dense implement of this algorithm, m_t, v_t and variable will update even if g is zero, but in sparse implement, m_t, v_t and variable will not update in iterations g is zero.

Args:

learning_rate: A Tensor or a floating point value. The learning rate. 
beta1: A float value or a constant float tensor. The exponential decay rate for the 1st moment estimates. 
beta2: A float value or a constant float tensor. The exponential decay rate for the 2nd moment estimates. 
epsilon: A small constant for numerical stability. 
use_locking: If True use locks for update operations. 

name: Optional name for the operations created when applying gradients. Defaults to “Adam”.

**2、在使用优化函数之前处理梯度**

使用minimize()操作，该操作不仅可以计算出梯度，而且还可以将梯度作用在变量上。如果想在使用它们之前处理梯度，可以按照以下三步骤使用optimizer ：

​       1、使用函数compute_gradients()计算梯度

​       2、按照自己的愿望处理梯度

​       3、使用函数apply_gradients()应用处理过后的梯度

```python
# 创建一个optimizer.
opt = GradientDescentOptimizer(learning_rate=0.1)
 
# 计算<list of variables>相关的梯度
grads_and_vars = opt.compute_gradients(loss, <list of variables>)
 
# grads_and_vars为tuples (gradient, variable)组成的列表。
#对梯度进行想要的处理，比如cap处理
capped_grads_and_vars = [(MyCapper(gv[0]), gv[1]) for gv in grads_and_vars]
 
# 令optimizer运用capped的梯度(gradients)
opt.apply_gradients(capped_grads_and_vars)
```

**3、选通梯度**

函数minimize() 与compute_gradients()都含有一个参数gate_gradient，用于控制在应用这些梯度时并行化的程度。

其值可以取：GATE_NONE, GATE_OP 或 GATE_GRAPH 
**GATE_NONE :** 并行地计算和应用梯度。提供最大化的并行执行，但是会导致有的数据结果没有再现性。比如两个matmul操作的梯度依赖输入值，使用GATE_NONE可能会出现有一个梯度在其他梯度之前便应用到某个输入中，导致出现不可再现的(non-reproducible)结果 
**GATE_OP:** 对于每个操作Op，确保每一个梯度在使用之前都已经计算完成。这种做法防止了那些具有多个输入，并且梯度计算依赖输入情形中，多输入Ops之间的竞争情况出现。 
**GATE_GRAPH:** 确保所有的变量对应的所有梯度在他们任何一个被使用前计算完成。该方式具有最低级别的并行化程度，但是对于想要在应用它们任何一个之前处理完所有的梯度计算时很有帮助的。

##### <2>Slots

一些optimizer的子类，比如 MomentumOptimizer 和 AdagradOptimizer 分配和管理着额外的用于训练的变量。这些变量称之为Slots，Slots有相应的名称，可以向optimizer访问的slots名称。有助于在log debug一个训练算法以及报告slots状态。

| 操作                                                         | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| tf.train.Optimizer.get_slot_names()                          | 返回一个由Optimizer所创建的slots的名称列表                   |
| tf.train.Optimizer.get_slot(var, name)                       | 返回一个name所对应的slot，name是由Optimizer为var所创建 var为用于传入 minimize() 或 apply_gradients()的变量 |
| class tf.train.GradientDescentOptimizer                      | 使用梯度下降算法的Optimizer                                  |
| tf.train.GradientDescentOptimizer.\__init__(learning_rate,  use_locking=False, name=’GradientDescent’) | 构建一个新的梯度下降优化器(Optimizer)                        |
| class tf.train.AdadeltaOptimizer                             | 使用Adadelta算法的Optimizer                                  |
| tf.train.AdadeltaOptimizer.\__init__(learning_rate=0.001,  rho=0.95, epsilon=1e-08,  use_locking=False, name=’Adadelta’) | 创建Adadelta优化器                                           |
| class tf.train.AdagradOptimizer                              | 使用Adagrad算法的Optimizer                                   |
| tf.train.AdagradOptimizer.\__init__(learning_rate,  initial_accumulator_value=0.1,  use_locking=False, name=’Adagrad’) | 创建Adagrad优化器                                            |
| class tf.train.MomentumOptimizer                             | 使用Momentum算法的Optimizer                                  |
| tf.train.MomentumOptimizer.\__init__(learning_rate,  momentum, use_locking=False,  name=’Momentum’, use_nesterov=False) | 创建momentum优化器 momentum：动量，一个tensor或者浮点值      |
| class tf.train.AdamOptimizer                                 | 使用Adam 算法的Optimizer                                     |
| tf.train.AdamOptimizer.\__init__(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name=’Adam’) | 创建Adam优化器                                               |
| class tf.train.FtrlOptimizer                                 | 使用FTRL 算法的Optimizer                                     |
| tf.train.FtrlOptimizer.\__init__(learning_rate,  learning_rate_power=-0.5,  initial_accumulator_value=0.1,  l1_regularization_strength=0.0,  l2_regularization_strength=0.0, use_locking=False, name=’Ftrl’) | 创建FTRL算法优化器                                           |
| class tf.train.RMSPropOptimizer                              | 使用RMSProp算法的Optimizer                                   |
| tf.train.RMSPropOptimizer.\__init__(learning_rate,  decay=0.9, momentum=0.0, epsilon=1e-10,  use_locking=False, name=’RMSProp’) | 创建RMSProp算法优化器                                        |

**tf.train.AdamOptimizer**

Adam 的基本运行方式，首先初始化：

```python
m_0 <- 0 (Initialize initial 1st moment vector)
v_0 <- 0 (Initialize initial 2nd moment vector)
t <- 0 (Initialize timestep)
```

在论文中的 section2 的末尾所描述了更新规则，该规则使用梯度g来更新变量：

```python
t <- t + 1
lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
 
m_t <- beta1 * m_{t-1} + (1 - beta1) * g
v_t <- beta2 * v_{t-1} + (1 - beta2) * g * g
variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)
```

其中epsilon 的默认值1e-8可能对于大多数情况都不是一个合适的值。例如，当在ImageNet上训练一个 Inception network时比较好的选择为1.0或者0.1。 
需要注意的是，在稠密数据中即便g为0时， m_t, v_t 以及variable都将会更新。而在稀疏数据中，m_t, v_t 以及variable不被更新且值为零。

##### <3>梯度计算与截断(Gradient Computation and Clipping)

TensorFlow 提供了计算给定tf计算图的求导函数，并在图的基础上增加节点。优化器(optimizer )类可以自动的计算网络图的导数，但是优化器中的创建器(creators )或者专业的人员可以通过本节所述的函数调用更底层的方法。

| 操作                                                         | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| tf.gradients(ys, xs, grad_ys=None, name=’gradients’,  colocate_gradients_with_ops=False, gate_gradients=False,  aggregation_method=None) | 构建一个符号函数，计算ys关于xs中x的偏导的和， 返回xs中每个x对应的sum(dy/dx) |
| tf.stop_gradient(input, name=None)                           | 停止计算梯度， 在EM算法、Boltzmann机等可能会使用到           |
| tf.clip_by_value(t, clip_value_min, clip_value_max, name=None) | 基于定义的min与max对tesor数据进行截断操作， 目的是为了应对梯度爆发或者梯度消失的情况 |
| tf.clip_by_norm(t, clip_norm, axes=None, name=None)          | 使用L2范式标准化tensor最大值为clip_norm 返回 t * clip_norm / l2norm(t) |
| tf.clip_by_average_norm(t, clip_norm, name=None)             | 使用平均L2范式规范tensor数据t， 并以clip_norm为最大值 返回 t * clip_norm / l2norm_avg(t) |
| tf.clip_by_global_norm(t_list,  clip_norm, use_norm=None, name=None) | 返回t_list[i] * clip_norm / max(global_norm, clip_norm) 其中global_norm = sqrt(sum([l2norm(t)**2 for t in t_list])) |
| tf.global_norm(t_list, name=None)                            | 返回global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))  |

##### <4>退化学习率(Decaying the learning rate)

| 操作                                                         | 描述                 |
| ------------------------------------------------------------ | -------------------- |
| tf.train.exponential_decay(learning_rate, global_step,  decay_steps, decay_rate, staircase=False, name=None) | 对学习率进行指数衰退 |

**tf.train.exponential_decay**

```python
#该函数返回以下结果
decayed_learning_rate = learning_rate *
         decay_rate ^ (global_step / decay_steps)
##例： 以0.96为基数，每100000 步进行一次学习率的衰退
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100000, 0.96, staircase=True)
# Passing global_step to minimize() will increment it at each step.
learning_step = (
    tf.train.GradientDescentOptimizer(learning_rate)
    .minimize(...my loss..., global_step=global_step)
)
```

##### <5>移动平均(Moving Averages)

一些训练优化算法，比如GradientDescent 和Momentum 在优化过程中便可以使用到移动平均方法。使用移动平均常常可以较明显地改善结果。

| 操作                                                         | 描述                         |
| ------------------------------------------------------------ | ---------------------------- |
| class tf.train.ExponentialMovingAverage                      | 将指数衰退加入到移动平均中   |
| tf.train.ExponentialMovingAverage.apply(var_list=None)       | 对var_list变量保持移动平均   |
| tf.train.ExponentialMovingAverage.average_name(var)          | 返回var均值的变量名称        |
| tf.train.ExponentialMovingAverage.average(var)               | 返回var均值变量              |
| tf.train.ExponentialMovingAverage.variables_to_restore(moving_avg_variables=None) | 返回用于保存的变量名称的映射 |

**tf.train.ExponentialMovingAverage**

```python
# Example usage when creating a training model:
# Create variables.
var0 = tf.Variable(...)
var1 = tf.Variable(...)
# ... use the variables to build a training model...
...
# Create an op that applies the optimizer.  This is what we usually
# would use as a training op.
opt_op = opt.minimize(my_loss, [var0, var1])
 
# Create an ExponentialMovingAverage object
ema = tf.train.ExponentialMovingAverage(decay=0.9999)
 
# Create the shadow variables, and add ops to maintain moving averages
# of var0 and var1.
maintain_averages_op = ema.apply([var0, var1])
 
# Create an op that will update the moving averages after each training
# step.  This is what we will use in place of the usual training op.
with tf.control_dependencies([opt_op]):
    training_op = tf.group(maintain_averages_op)
 
...train the model by running training_op...
 
#Example of restoring the shadow variable values:
# Create a Saver that loads variables from their saved shadow values.
shadow_var0_name = ema.average_name(var0)
shadow_var1_name = ema.average_name(var1)
saver = tf.train.Saver({shadow_var0_name: var0, shadow_var1_name: var1})
saver.restore(...checkpoint filename...)
# var0 and var1 now hold the moving average values
```

**tf.train.ExponentialMovingAverage.variables_to_restore**

```python
variables_to_restore = ema.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)
```

##### <6>协调器和队列运行器

查看queue中，queue相关的内容，了解tensorflow中队列的运行方式。

| 操作                                                         | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| class tf.train.Coordinator                                   | 线程的协调器                                                 |
| tf.train.Coordinator.clear_stop()                            | 清除停止标记                                                 |
| tf.train.Coordinator.join(threads=None, stop_grace_period_secs=120) | 等待线程终止 threads:一个threading.Threads的列表，启动的线程，将额外加入到registered的线程中 |
| tf.train.Coordinator.register_thread(thread)                 | Register一个用于join的线程                                   |
| tf.train.Coordinator.request_stop(ex=None)                   | 请求线程结束                                                 |
| tf.train.Coordinator.should_stop()                           | 检查是否被请求停止                                           |
| tf.train.Coordinator.stop_on_exception()                     | 上下文管理器，当一个例外出现时请求停止                       |
| tf.train.Coordinator.wait_for_stop(timeout=None)             | 等待Coordinator提示停止进程                                  |
| class tf.train.QueueRunner                                   | 持有一个队列的入列操作列表，用于线程中运行 queue:一个队列 enqueue_ops: 用于线程中运行的入列操作列表 |
| tf.train.QueueRunner.create_threads(sess,  coord=None, daemon=False, start=False) | 创建运行入列操作的线程，返回一个线程列表                     |
| tf.train.QueueRunner.from_proto(queue_runner_def)            | 返回由queue_runner_def创建的QueueRunner对象                  |
| tf.train.add_queue_runner(qr, collection=’queue_runners’)    | 增加一个QueueRunner到graph的收集器(collection )中            |
| tf.train.start_queue_runners(sess=None, coord=None, daemon=True, start=True, collection=’queue_runners’) | 启动所有graph收集到的队列运行器(queue runners)               |

**class tf.train.Coordinator**

```python
#Coordinator的使用，用于多线程的协调
try:
  ...
  coord = Coordinator()
  # Start a number of threads, passing the coordinator to each of them.
  ...start thread 1...(coord, ...)
  ...start thread N...(coord, ...)
  # Wait for all the threads to terminate, give them 10s grace period
  coord.join(threads, stop_grace_period_secs=10)
except RuntimeException:
  ...one of the threads took more than 10s to stop after request_stop()
  ...was called.
except Exception:
  ...exception that was passed to coord.request_stop()
```

**tf.train.Coordinator.stop_on_exception()**

```python
with coord.stop_on_exception():
  # Any exception raised in the body of the with
  # clause is reported to the coordinator before terminating
  # the execution of the body.
  ...body...
#等价于
try:
  ...body...
exception Exception as ex:
  coord.request_stop(ex)
```

##### <7>分布执行

可以阅读[TensorFlow的分布式学习框架简介 ](http://blog.csdn.net/lenbow/article/details/52130565)查看更多tensorflow分布式细节。

| 操作                                                         | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| class tf.train.Server                                        | 一个进程内的tensorflow服务，用于分布式训练                   |
| tf.train.Server.init(server_or_cluster_def,  job_name=None, task_index=None, protocol=None, config=None, start=True) | 创建一个新的服务，其中job_name, task_index,  和protocol为可选参数， 优先级高于server_or_cluster_def中相关信息 server_or_cluster_def : 为一个tf.train.ServerDef  或 tf.train.ClusterDef 协议(protocol)的buffer， 或者一个tf.train.ClusterSpec对象 |
| tf.train.Server.create_local_server(config=None, start=True) | 创建一个新的运行在本地主机的单进程集群                       |
| tf.train.Server.target                                       | 返回tf.Session所连接的目标服务器                             |
| tf.train.Server.server_def                                   | 返回该服务的tf.train.ServerDef                               |
| tf.train.Server.start()                                      | 开启服务                                                     |
| tf.train.Server.join()                                       | 阻塞直到服务已经关闭                                         |

| #                                                            |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| class tf.train.Supervisor                                    | 一个训练辅助器，用于checkpoints模型以及计算的summaries。该监视器只是一个小的外壳(wrapper),用于Coordinator, a Saver, 和a SessionManager周围 |
| tf.train.Supervisor.__init__(graph=None, ready_op=0, is_chief=True, init_op=0, init_feed_dict=None, local_init_op=0, logdir=None,  summary_op=0, saver=0, global_step=0,  save_summaries_secs=120, save_model_secs=600,  recovery_wait_secs=30, stop_grace_secs=120, checkpoint_basename=’model.ckpt’, session_manager=None, summary_writer=0, init_fn=None) | 创建一个监视器Supervisor                                     |
| tf.train.Supervisor.managed_session(master=”, config=None, start_standard_services=True, close_summary_writer=True) | 返回一个管路session的上下文管理器                            |
| tf.train.Supervisor.prepare_or_wait_for_session(master=”, config=None, wait_for_checkpoint=False, max_wait_secs=7200, start_standard_services=True) | 确保model已经准备好                                          |
| tf.train.Supervisor.start_standard_services(sess)            | 为sess启动一个标准的服务                                     |
| tf.train.Supervisor.start_queue_runners(sess, queue_runners=None) | 为QueueRunners启动一个线程，queue_runners为一个QueueRunners列表 |
| tf.train.Supervisor.summary_computed(sess, summary, global_step=None) | 指示计算的summary                                            |
| tf.train.Supervisor.stop(threads=None, close_summary_writer=True) | 停止服务以及协调器(coordinator),并没有关闭session            |
| tf.train.Supervisor.request_stop(ex=None)                    | 参考Coordinator.request_stop()                               |
| tf.train.Supervisor.should_stop()                            | 参考Coordinator.should_stop()                                |
| tf.train.Supervisor.stop_on_exception()                      | 参考 Coordinator.stop_on_exception()                         |
| tf.train.Supervisor.Loop(timer_interval_secs, target, args=None, kwargs=None) | 开启一个循环器线程用于调用一个函数 每经过timer_interval_secs秒执行，target(*args, **kwargs) |
| tf.train.Supervisor.coord                                    | 返回监督器(Supervisor)使用的协调器(Coordinator )             |

| #                                                            |                                                           |
| ------------------------------------------------------------ | --------------------------------------------------------- |
| class tf.train.SessionManager                                | 训练的辅助器，用于从checkpoint恢复数据以及创建一个session |
| tf.train.SessionManager.__init__(local_init_op=None, ready_op=None, graph=None, recovery_wait_secs=30) | 创建一个SessionManager                                    |
| tf.train.SessionManager.prepare_session(master, init_op=None, saver=None, checkpoint_dir=None, wait_for_checkpoint=False, max_wait_secs=7200, config=None, init_feed_dict=None, init_fn=None) | 创建一个session，并确保model可以被使用                    |
| tf.train.SessionManager.recover_session(master, saver=None, checkpoint_dir=None, wait_for_checkpoint=False, max_wait_secs=7200, config=None) | 创建一个session，如果可以的话，使用恢复方法创建           |
| tf.train.SessionManager.wait_for_session(master, config=None, max_wait_secs=inf) | 创建一个session，并等待model准备完成                      |

| #                                                            |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| class tf.train.ClusterSpec                                   | 将一个集群表示为一系列“tasks”，并整合至“jobs”中              |
| tf.train.ClusterSpec.as_cluster_def()                        | 返回该cluster中一个tf.train.ClusterDef协议的buffer           |
| tf.train.ClusterSpec.as_dict()                               | 返回一个字典，由job名称对应于网络地址                        |
| tf.train.ClusterSpec.job_tasks(job_name)                     | 返回一个给定的job对应的task列表                              |
| tf.train.ClusterSpec.jobs                                    | 返回该cluster的job名称列表                                   |
| tf.train.replica_device_setter(ps_tasks=0, ps_device=’/job:ps’, worker_device=’/job:worker’, merge_devices=True, cluster=None, ps_ops=None) | 返回一个设备函数(device function)，以在建立一个副本graph的时候使用，设备函数(device function)用在with tf.device(device_function)中 |

**tf.train.Server**

```python
server = tf.train.Server(...)
with tf.Session(server.target):
    #...
```

**tf.train.Supervisor** 

相关参数： 
ready_op : 一维 字符串 tensor。该tensor是用过监视器在prepare_or_wait_for_session()计算，检查model是否准备好可以使用。如果准备好，将返回一个空阵列，如果为None，该model没有被检查。 
is_chief : 如果为True，创建一个主监视器用于负责初始化与模型的恢复，若为False，则依赖主监视器。 
init_op : 一个操作，用于模型不能恢复时的初始化操作。默认初始化所有操作 
local_init_op : 可被所有监视器运行的初始化操作。 
logdir : 设置log目录 
summary_op : 一个操作(Operation )，返回Summary 和事件logs，需要设置 logdir 
saver : 一个Saver对象 
save_summaries_secs : 保存summaries的间隔秒数 
save_model_secs : 保存model的间隔秒数 
checkpoint_basename : checkpoint保存的基本名称

- 使用在单进程中

```python

with tf.Graph().as_default():
  ...add operations to the graph...
  # Create a Supervisor that will checkpoint the model in '/tmp/mydir'.
  sv = Supervisor(logdir='/tmp/mydir')
  # Get a TensorFlow session managed by the supervisor.
  with sv.managed_session(FLAGS.master) as sess:
    # Use the session to train the graph.
    while not sv.should_stop():
      sess.run(<my_train_op>)
# 在上下文管理器with sv.managed_session()内，所有在graph的变量都被初始化。
# 或者说，一些服务器checkpoint相应模型并增加summaries至事件log中。
# 如果有例外发生，should_stop()将返回True
```

- 使用在多副本运行情况中 
  要使用副本训练已经部署在集群上的相同程序，必须指定其中一个task为主要，该task处理 initialization, checkpoints, summaries, 和recovery相关事物。其他task依赖该task。

```python
# Choose a task as the chief. This could be based on server_def.task_index,
# or job_def.name, or job_def.tasks. It's entirely up to the end user.
# But there can be only one *chief*.
is_chief = (server_def.task_index == 0)
server = tf.train.Server(server_def)
 
with tf.Graph().as_default():
  ...add operations to the graph...
  # Create a Supervisor that uses log directory on a shared file system.
  # Indicate if you are the 'chief'
  sv = Supervisor(logdir='/shared_directory/...', is_chief=is_chief)
  # Get a Session in a TensorFlow server on the cluster.
  with sv.managed_session(server.target) as sess:
    # Use the session to train the graph.
    while not sv.should_stop():
      sess.run(<my_train_op>)
```

如果有task崩溃或重启，managed_session() 将检查是否Model被初始化。如果已经初始化，它只需要创建一个session并将其返回至正在训练的正常代码中。如果model需要被初始化，主task将对它进行重新初始化，而其他task将等待模型初始化完成。 
注意：该程序方法一样适用于单进程的work，该单进程标注自己为主要的便行

**supervisor中master的字符串形式** 

无论运行在本机或者集群上，都可以使用以下值设定master flag：

- 定义为 ” ，要求一个进程内且没有使用RPC的session
- 定义为 ‘local’，要求一个使用基于RPC的主服务接口(“Master interface” )的session来运行tensorflow程序。更多细节可以查看 tf.train.Server.create_local_server()相关内容。
- 定义为 ‘grpc://hostname:port’，要求一个指定的RPC接口的session，同时运行内部进程的master接入远程的tensorflow workers。可用server.target返回该形式

 **supervisor高级用法**

- 启动额外的服务 
  managed_session()启动了 Checkpoint 和Summary服务。如果需要运行更多的服务，可以在managed_session()控制的模块中启动他们。

```python
#例如： 开启一个线程用于打印loss. 设置每60秒该线程运行一次，我们使用sv.loop()
 ...
  sv = Supervisor(logdir='/tmp/mydir')
  with sv.managed_session(FLAGS.master) as sess:
    sv.loop(60, print_loss, (sess))
    while not sv.should_stop():
      sess.run(my_train_op)
```

- 启动更少的的服务 

managed_session() 启动了 “summary” 和 “checkpoint” 线程，这些线程通过构建器或者监督器默认自动创建了summary_op 和saver操作。如果想运行自己的 summary 和checkpointing方法，关闭这些服务，通过传递None值给summary_op 和saver参数。

```python
在chief中每100个step，创建summaries
  # Create a Supervisor with no automatic summaries.
  sv = Supervisor(logdir='/tmp/mydir', is_chief=is_chief, summary_op=None)
  # As summary_op was None, managed_session() does not start the
  # summary thread.
  with sv.managed_session(FLAGS.master) as sess:
    for step in xrange(1000000):
      if sv.should_stop():
        break
      if is_chief and step % 100 == 0:
        # Create the summary every 100 chief steps.
        sv.summary_computed(sess, sess.run(my_summary_op))
      else:
        # Train normally
        sess.run(my_train_op)

```

**tf.train.Supervisor.managed_session**

```python
def train():
  sv = tf.train.Supervisor(...)
  with sv.managed_session(<master>) as sess:
    for step in xrange(..):
      if sv.should_stop():
        break
      sess.run(<my training op>)
      ...do other things needed at each training step...
```

**tf.train.SessionManager**

```python
with tf.Graph().as_default():
   ...add operations to the graph...
  # Create a SessionManager that will checkpoint the model in '/tmp/mydir'.
  sm = SessionManager()
  sess = sm.prepare_session(master, init_op, saver, checkpoint_dir)
  # Use the session to train the graph.
  while True:
    sess.run(<my_train_op>)
#其中prepare_session()初始化和恢复一个模型参数。 
 
#另一个进程将等待model准备完成，代码如下
with tf.Graph().as_default():
  ...add operations to the graph...
  # Create a SessionManager that will wait for the model to become ready.
  sm = SessionManager()
  sess = sm.wait_for_session(master)
  # Use the session to train the graph.
  while True:
    sess.run(<my_train_op>)
#wait_for_session()等待一个model被其他进程初始化
```

**tf.train.ClusterSpec** 

一个tf.train.ClusterSpec表示一系列的进程，这些进程都参与分布式tensorflow的计算。每一个 tf.train.Server都在一个独有的集群中构建。 
创建一个具有两个jobs及其5个tasks的集群们需要定义从job名称列表到网络地址列表之间的映射。

```python
cluster = tf.train.ClusterSpec({"worker": ["worker0.example.com:2222",
                                           "worker1.example.com:2222",
                                           "worker2.example.com:2222"],
                                "ps": ["ps0.example.com:2222",
                                       "ps1.example.com:2222"]})
```

**tf.train.replica_device_setter**

```python
# To build a cluster with two ps jobs on hosts ps0 and ps1, and 3 worker
# jobs on hosts worker0, worker1 and worker2.
cluster_spec = {
    "ps": ["ps0:2222", "ps1:2222"],
    "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]}
with tf.device(tf.replica_device_setter(cluster=cluster_spec)):
  # Build your graph
  v1 = tf.Variable(...)  # assigned to /job:ps/task:0
  v2 = tf.Variable(...)  # assigned to /job:ps/task:1
  v3 = tf.Variable(...)  # assigned to /job:ps/task:0
# Run compute
```

##### <8>汇总操作

我们可以在一个session中获取summary操作的输出，并将其传输到SummaryWriter以添加至一个事件记录文件中。

| 操作                                                         | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| tf.scalar_summary(tags, values, collections=None, name=None) | 输出一个标量值的summary协议buffer tag的shape需要与values的相同，用来做summaries的tags，为字符串 |
| tf.image_summary(tag, tensor, max_images=3, collections=None, name=None) | 输出一个图像tensor的summary协议buffer                        |
| tf.audio_summary(tag, tensor, sample_rate, max_outputs=3, collections=None, name=None) | 输出一个音频tensor的summary协议buffer                        |
| tf.histogram_summary(tag, values, collections=None, name=None) | 输出一个直方图的summary协议buffer                            |
| tf.nn.zero_fraction(value, name=None)                        | 返回0在value中的小数比例                                     |
| tf.merge_summary(inputs, collections=None, name=None)        | 合并summary                                                  |
| tf.merge_all_summaries(key=’summaries’)                      | 合并在默认graph中手机的summaries                             |

**将记录汇总写入文件中(Adding Summaries to Event Files)**

| 操作                                                         | 描述                                            |
| ------------------------------------------------------------ | ----------------------------------------------- |
| class tf.train.SummaryWriter                                 | 将summary协议buffer写入事件文件中               |
| tf.train.SummaryWriter.__init__(logdir, graph=None, max_queue=10, flush_secs=120, graph_def=None) | 创建一个SummaryWriter实例以及新建一个事件文件   |
| tf.train.SummaryWriter.add_summary(summary, global_step=None) | 将一个summary添加到事件文件中                   |
| tf.train.SummaryWriter.add_session_log(session_log, global_step=None) | 添加SessionLog到一个事件文件中                  |
| tf.train.SummaryWriter.add_event(event)                      | 添加一个事件到事件文件中                        |
| tf.train.SummaryWriter.add_graph(graph, global_step=None, graph_def=None) | 添加一个Graph到时间文件中                       |
| tf.train.SummaryWriter.add_run_metadata(run_metadata, tag, global_step=None) | 为一个单一的session.run()调用添加一个元数据信息 |
| tf.train.SummaryWriter.flush()                               | 刷新时间文件到硬盘中                            |
| tf.train.SummaryWriter.close()                               | 将事件问价写入硬盘中并关闭该文件                |
| tf.train.summary_iterator(path)                              | 一个用于从时间文件中读取时间协议buffer的迭代器  |

 **tf.train.SummaryWriter** 

创建一个SummaryWriter 和事件文件。如果我们传递一个Graph进入该构建器中，它将被添加到事件文件当中，这一点与使用add_graph()具有相同功能。 
TensorBoard 将从事件文件中提取该graph，并将其显示。所以我们能直观地看到我们建立的graph。我们通常从我们启动的session中传递graph：

```python
...create a graph...
# Launch the graph in a session.
sess = tf.Session()
# Create a summary writer, add the 'graph' to the event file.
writer = tf.train.SummaryWriter(<some-directory>, sess.graph)
```

**tf.train.summary_iterator**

```python
#打印时间文件中的内容
for e in tf.train.summary_iterator(path to events file):
    print(e)
 
#打印指定的summary值
# This example supposes that the events file contains summaries with a
# summary value tag 'loss'.  These could have been added by calling
# `add_summary()`, passing the output of a scalar summary op created with
# with: `tf.scalar_summary(['loss'], loss_tensor)`.
for e in tf.train.summary_iterator(path to events file):
    for v in e.summary.value:
        if v.tag == 'loss':
            print(v.simple_value)
```

##### <9>训练的通用函数及其他

| 操作                                                         | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| tf.train.global_step(sess, global_step_tensor)               | 一个用于获取全局step的小辅助器                               |
| tf.train.write_graph(graph_def, logdir, name, as_text=True)  | 将一个graph proto写入一个文件中                              |
| #                                                            |                                                              |
|                                                              | :—                                                           |
| class tf.train.LooperThread                                  | 可重复地执行代码的线程                                       |
| tf.train.LooperThread.init(coord, timer_interval_secs, target=None, args=None, kwargs=None) | 创建一个LooperThread                                         |
| tf.train.LooperThread.is_alive()                             | 返回是否该线程是活跃的                                       |
| tf.train.LooperThread.join(timeout=None)                     | 等待线程结束                                                 |
| tf.train.LooperThread.loop(coord, timer_interval_secs, target, args=None, kwargs=None) | 启动一个LooperThread，用于周期地调用某个函数 调用函数target(args) |
| tf.py_func(func, inp, Tout, stateful=True, name=None)        | 将python函数包装成tf中操作节点                               |

**tf.train.global_step**

```python
# Creates a variable to hold the global_step.
global_step_tensor = tf.Variable(10, trainable=False, name='global_step')
# Creates a session.
sess = tf.Session()
# Initializes the variable.
sess.run(global_step_tensor.initializer)
print('global_step: %s' % tf.train.global_step(sess, global_step_tensor))
 
global_step: 10
```

**tf.train.write_graph**

```python
v = tf.Variable(0, name='my_variable')
sess = tf.Session()
tf.train.write_graph(sess.graph_def, '/tmp/my-model', 'train.pbtxt')
```

**tf.py_func**

```python
#tf.py_func(func, inp, Tout, stateful=True, name=None)
#func：为一个python函数
#inp：为输入函数的参数，
#Tout： 指定func返回的输出的数据类型，是一个列表
def my_func(x):
  # x will be a numpy array with the contents of the placeholder below
  return np.sinh(x)
inp = tf.placeholder(tf.float32, [...])
y = py_func(my_func, [inp], [tf.float32])
```

#### 8、测试函数

TensorFlow 提供了一个方便的继承unittest.TestCase类的方法，该类增加有关TensorFlow 测试的方法。如下例子：

```python
import tensorflow as tf
 
class SquareTest(tf.test.TestCase):
 
  def testSquare(self):
    with self.test_session():
      x = tf.square([2, 3])
      self.assertAllEqual(x.eval(), [4, 9])
 
 
if __name__ == '__main__':
  tf.test.main()
```

##### <1>共用（Utilities）

| 操作                                             | 描述                                   |
| ------------------------------------------------ | -------------------------------------- |
| tf.test.main()                                   | 运行所有的单元测试                     |
| tf.test.assert_equal_graph_def(actual, expected) | 断言 两个GraphDefs 是否几乎一样        |
| tf.test.get_temp_dir()                           | 返回测试期间使用的临时目录             |
| tf.test.is_built_with_cuda()                     | 返回是否Tensorflow支持CUDA(GPU)的build |

##### <2>梯度检查(Gradient checking)

可对比compute_gradient 和compute_gradient_error函数的用法

| 操作                                                         | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| tf.test.compute_gradient(x, x_shape, y, y_shape, x_init_value=None, delta=0.001, init_targets=None) | 计算并返回理论的和数值的Jacobian矩阵                         |
| tf.test.compute_gradient_error(x, x_shape, y, y_shape, x_init_value=None, delta=0.001, init_targets=None) | 计算梯度的error。在计算所得的与数值估计的Jacobian中 为dy/dx计算最大的error |



#### 9、符号嵌入

##### <1>tf.nn.embedding_lookup

tf.nn.embedding_lookup(params, ids, partition_strategy=’mod’, name=None, validate_indices=True)

根据索引ids查询embedding列表params中的tensor值

如果len(params) >1，id将会安照partition_strategy策略进行分割

1、如果partition_strategy为”mod”，id所分配到的位置为p = id % len(params)

比如有13个ids，分为5个位置，那么分配方案为：[[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8], [4, 9]]

2、如果partition_strategy为”div”,那么分配方案为：[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]]

##### <2>tf.nn.embedding_lookup_sparse

tf.nn.embedding_lookup_sparse(params, sp_ids, sp_weights, partition_strategy=’mod’, name=None, combiner=’mean’)

对给定的ids和权重查询embedding

1、sp_ids为一个N x M的稀疏tensor，N为batch大小，M为任意，数据类型int64

2、sp_weights的shape与sp_ids的稀疏tensor权重，浮点类型，若为None，则权重为全’1’

#### 10、循环神经网络

##### <1>tf.nn.rnn

tf.nn.rnn(cell, inputs, initial_state=None, dtype=None, sequence_length=None, scope=None)

基于RNNCell类的实例cell建立循环神经网络

##### <2>tf.nn.dynamic_rnn

tf.nn.dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None, dtype=None, parallel_iterations=None, swap_memory=False, time_major=False, scope=None)

基于RNNCell类的实例cell建立动态循环神经网络

与一般rnn不同的是，该函数会根据输入动态展开返回**(outputs,state)**

为了描述输出的形状，先介绍几个变量，batch_size是输入的这批数据的数量，max_time就是这批数据中序列的最长长度，如果输入的三个句子，那max_time对应的就是最长句子的单词数量，cell.output_size其实就是rnn cell中神经元的个数。

**outputs：**outputs是一个tensor

- 如果time_major==True，outputs形状为 [max_time, batch_size, cell.output_size ]（要求rnn输入与rnn输出形状保持一致）

- 如果time_major==False（默认），outputs形状为 [ batch_size, max_time, cell.output_size ]

**state：**state是一个tensor

state是最终的状态，也就是序列中最后一个cell输出的状态。一般情况下state的形状为 [batch_size, cell.output_size ]。

但当输入的cell为BasicLSTMCell时，state的形状为[2，batch_size, cell.output_size ]，其中2也对应着LSTM中的cell state和hidden state。

**output记录每个序列的点的输出，state的只记录序列最终的输出**

##### <3>tf.nn.state_saving_rnn

tf.nn.state_saving_rnn(cell, inputs, state_saver, state_name, sequence_length=None, scope=None)

可储存调试状态的RNN网络

##### <4>tf.nn.bidirectional_rnn

tf.nn.bidirectional_rnn(cell_fw, cell_bw, inputs, initial_state_fw=None, initial_state_bw=None, dtype=None,sequence_length=None, scope=None)

双向RNN, 返回一个3元组tuple(outputs, output_state_fw, output_state_bw)

#### 11、求值网络

##### <1>tf.nn.top_k

tf.nn.top_k(input, k=1, sorted=True, name=None)

返回前k大的值及其对应的索引

##### <2>tf.nn.in_top_k

tf.nn.in_top_k(predictions, targets, k,name=None)

返回判断是否targets索引的predictions相应的值是否在predictions前k个位置中

返回数据类型为bool类型，len与predictions同

#### 12、监督候选采样网络

对于有巨大量的多分类与多标签模型，如果使用全连接softmax将会占用大量的时间与空间资源，所以采用候选采样方法仅使用一小部分类别与标签作为监督以加速训练。

##### <1>tf.nn.nce_loss

tf.nn.nce_loss(weights, biases, inputs,labels,num_sampled,num_classes,num_true=1,sampled_values=None,

remove_accidental_hits=False, partition_strategy=’mod’,name=’nce_loss’)

返回noise-contrastive的训练损失结果

##### <2>tf.nn.sampled_softmax_loss

tf.nn.sampled_softmax_loss(weights, biases, inputs, labels, num_sampled, num_classes, num_true=1, sampled_values=None,remove_accidental_hits=True,partition_strategy=’mod’,name=’sampled_softmax_loss’)

返回sampled softmax的训练损失

##### <3>tf.nn.uniform_candidate_sampler

tf.nn.uniform_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, seed=None, name=None)

通过均匀分布的采样集合

返回三元tuple

1、sampled_candidates 候选集合。

2、期望的true_classes个数，为浮点值

3、期望的sampled_candidates个数，为浮点值

##### <4>tf.nn.log_uniform_candidate_sampler

tf.nn.log_uniform_candidate_sampler(true_classes, num_true,num_sampled, unique, range_max, seed=None, name=None)

通过log均匀分布的采样集合，返回三元tuple

##### <5>tf.nn.learned_unigram_candidate_sampler

tf.nn.learned_unigram_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, seed=None, name=None)

根据在训练过程中学习到的分布状况进行采样

返回三元tuple

##### <6>tf.nn.fixed_unigram_candidate_sampler

tf.nn.fixed_unigram_candidate_sampler(true_classes, num_true,num_sampled, unique, range_max, vocab_file=”, distortion=1.0, num_reserved_ids=0, num_shards=1, shard=0, unigrams=(), seed=None, name=None)

基于所提供的基本分布进行采样



------





### （9）保存与恢复变量

##### <1>tf.train.Saver.\__init__

tf.train.Saver.\__init__(var_list=None, reshape=False, sharded=False, max_to_keep=5, keep_checkpoint_every_n_hours=10000.0, name=None, restore_sequentially=False,saver_def=None, builder=None)

创建一个存储器Saver

var_list定义需要存储和恢复的变量

##### <2>tf.train.Saver.save

tf.train.Saver.save(sess,save_path,global_step=None,latest_filename=None,meta_graph_suffix=’meta’,write_meta_graph=True)

保存变量

##### <3>tf.train.Saver.restore

tf.train.Saver.restore(sess, save_path)

恢复变量

##### <4>tf.train.Saver.last_checkpoints

列出最近未删除的checkpoint文件名

##### <5>tf.train.Saver.set_last_checkpoints

tf.train.Saver.set_last_checkpoints(last_checkpoints)

设置checkpoint文件名列表

##### <6>tf.train.Saver.set_last_checkpoints_with_time

 tf.train.Saver.set_last_checkpoints_with_time(last_checkpoints_with_time)

 设置checkpoint文件名列表和时间戳   

### （10）关于numpy引用的函数

##### <1>np.linspace

np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)

此函数作用为：在规定的时间内，返回固定间隔的数据。他将返回“num”个等间距的样本，在区间[`start`, `stop`]中。其中，区间的结束端点可以被排除在外。。

**参数：**

**start：**scalar（标量） 序列的起始点

**stop：**序列的结束点，当endpoint=False时,不包含该点。在这种情况下，该序列包括除了num+1以外的所有等间距的样本，以致于stop被排除。当endpoint为False时注意步长会发生改变。

**num：**int类型，为可选参数，表示生成的样本数，默认是50。必须是非负数。

**endpoint：**bool类型，为可选参数，如果是True，stop是最后的样本。否则，stop将不会被包含。默认为true

**retstep：**返回值形式，默认为False，返回等差数列；若为Ture，则返回结果（array([samples,step]))

**dtype：**返回结果的数据类型，默认没有，若没有的话，则参考输入数据类型

**返回值：**

**samples：**ndarray（多维数组）

在闭区间[start, stop]或者是半开区间[start, stop)中有num个等间距的样本

**step：**float

仅仅当retstep=True时候会返回。样本之间间距的大小

##### <2>np.newaxis

它的功能是插入新维度。

**例：**

```python
a=np.array([1,2,3,4,5])
print a.shape
print a
```

输出结果：

(5,)
[1 2 3 4 5]

可以看出a是一个一维数组

```python
a=np.array([1,2,3,4,5])
b=a[np.newaxis,:]
print a.shape,b.shape
print a
print b
```

输出结果：

(5,) (1, 5)
[1 2 3 4 5]
[[1 2 3 4 5]]

```python
a=np.array([1,2,3,4,5])
b=a[:,np.newaxis]
print a.shape,b.shape
print a
print b
```

输出结果:

(5,) (5, 1)
[1 2 3 4 5]
[[1]
 [2]
 [3]
 [4]
 [5]]

可以看出np.newaxis分别是在行或列上增加维度，原来是（5，）的数组，在行上增加维度变成（1,5）的二维数组，在列上增加维度变为(5,1)的二维数组。

##### <3>np.random.normal

numpy.random.normal(loc,scale,size=shape)

输出正态分布

**参数：**

**loc：**正态分布的均值，对应这个分布的中心。

**scale：**正态分布的标准差，对应分布的宽度。scale越大，正态分布的曲线越矮胖，scale越小，曲线越瘦高。

**size：**输出的值赋在shape里，默认为None。

##### <4>np.random.randn()

**语法：**

np.random.randn(d0,d1,d2……dn) 

1 ) 当函数括号内没有参数时，则返回一个浮点数； 

2）当函数括号内有一个参数时，则返回秩为1的数组，不能表示向量和矩阵； 

3）当函数括号内有两个及以上参数时，则返回对应维度的数组，能表示向量或矩阵； 

4）np.random.standard_normal（）函数与np.random.randn()类似，但是np.random.standard_normal（）的输入参数为元组（tuple）. 

5 ) np.random.randn()的输入通常为整数，但是如果为浮点数，则会自动直接截断转换为整数。

**作用：**

通过本函数可以返回一个或一组服从标准正态分布的随机样本值。（标准正态分布前面已有讲解）

**示例：**

![](/random示例1.png)



**应用场景：**

在神经网络构建中，权重参数W通常采用该函数进行初始化，当然需要注意的是，通常会在生成的矩阵后面乘以小数，比如0.01，目的是为了提高梯度下降算法的收敛速度。 
W = np.random.randn(2,2)*0.01

##### <5>np.random.rand()

**语法：**

np.random.rand(d0,d1,d2……dn)
注：使用方法与np.random.randn()函数相同 

**作用：**

通过本函数可以返回一个或一组服从**“0~1”均匀分布**的随机样本值。随机样本取值范围是[0,1)，不包括1。 

**示例：**

![](/random示例2.png)

**应用场景：**

在深度学习的Dropout正则化方法中，可以用于生成dropout随机向量（dl），例如（keep_prob表示保留神经元的比例）：dl = np.random.rand(al.shape[0],al.shape[1]) < keep_prob

##### <6>np.random.randint()

**语法：**

numpy.random.randint(low, high=None, size=None, dtype=’l’)

输入： 
low—–为最小值 
high—-为最大值 
size—–为数组维度大小 
dtype—为数据类型，默认的数据类型是np.int。 
返回值： 
返回随机整数或整型数组，范围区间为[low,high），包含low，不包含high； 

high没有填写时，默认生成随机数的范围是[0，low）

**示例：**

![](/randm示例3.png)



### （11）、tensorboard相关

tensorboard 作为一款可视化神器，可以说是学习tensorflow时模型训练以及参数可视化的法宝。

而在训练过程中，主要用到了tf.summary()的各类方法，能够保存训练过程以及参数分布图并在tensorboard显示。

##### <1>tf.summary.scalar

tf.summary.scalar(name,tensor,collections=None,family=None)

用来显示标量信息，一般在画loss,accuary时会用到这个函数。

**参数：**

- **name**:生成节点的名字，也会作为TensorBoard中的系列的名字。
- **tensor**:包含一个值的实数Tensor。
- **collection**：图的集合键值的可选列表。新的求和op被添加到这个集合中。缺省为`[GraphKeys.SUMMARIES]`
- **family**:可选项；设置时用作求和标签名称的前缀，这影响着TensorBoard所显示的标签名。

**返回值：**

- **返回值**：一个字符串类型的标量张量，包含一个`Summary`protobuf
- **返回错误**：`ValueError`tensor有错误的类型或shape。
- 函数求出的`Summary`中有一个包含输入tensor的Tensor.proto

##### <2>tf.summary.histogram

用来显示直方图信息，其格式为：

tf.summary.histogram(tags, values, collections=None, name=None) 

例如： tf.summary.histogram('histogram', var)

一般用来显示训练过程中变量的分布情况

##### <3>tf.summary.distribution

分布图，一般用于显示weights分布

##### <4>tf.summary.text

可以将文本类型的数据转换为tensor写入summary中：

例如：

text = """/a/b/c\\_d/f\\_g\\_h\\_2017"""
summary_op0 = tf.summary.text('text', tf.convert_to_tensor(text))

##### <5>tf.summary.image

输出带图像的probuf，汇总数据的图像的的形式如下： ' tag /image/0', ' tag /image/1'...，如：input/image/0等。

格式：tf.summary.image(tag, tensor, max_images=3, collections=None, name=Non

##### <6>tf.summary.audio

展示训练过程中记录的音频 

##### <7>tf.summary.merge_all

merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示。如果没有特殊要求，一般用这一句就可

以显示训练时的各种信息了。

格式：tf.summaries.merge_all(key='summaries')

##### <8>tf.summary.FileWriter

指定一个文件用来保存图。

格式：tf.summary.FileWritter(path,sess.graph)

可以调用其add_summary（）方法将训练过程数据保存在filewriter指定的文件中

Tensorflow Summary 用法示例:

```python
tf.summary.scalar('accuracy',acc)                   #生成准确率标量图  
merge_summary = tf.summary.merge_all()  
train_writer = tf.summary.FileWriter(dir,sess.graph)#定义一个写入summary的目标文件，dir为写入文件地址  
......(交叉熵、优化器等定义)  
for step in xrange(training_step):                  #训练循环  
    train_summary = sess.run(merge_summary,feed_dict =  {...})#调用sess.run运行图，生成一步的训练过程数据  
    train_writer.add_summary(train_summary,step)#调用train_writer的add_summary方法将训练过程以及训练步数保存 
```

此时开启tensorborad：  tensorboard --logdir=/summary_dir

便能看见accuracy曲线了。

另外，如果我不想保存所有定义的summary信息，也可以用tf.summary.merge方法有选择性地保存信息

##### <9>tf.summary.merge

格式：tf.summary.merge(inputs, collections=None, name=None)

一般选择要保存的信息还需要用到tf.get_collection()函数

示例：

```python
tf.summary.scalar('accuracy',acc)                   #生成准确率标量图  
merge_summary = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES,'accuracy'),...(其他要显示的信息)])  
train_writer = tf.summary.FileWriter(dir,sess.graph)#定义一个写入summary的目标文件，dir为写入文件地址  
......(交叉熵、优化器等定义)  
for step in xrange(training_step):                  #训练循环  
    train_summary = sess.run(merge_summary,feed_dict =  {...})#调用sess.run运行图，生成一步的训练过程数据  
    train_writer.add_summary(train_summary,step)#调用train_writer的add_summary方法将训练过程以及训练步数保存 
```

当然，也可以直接：

```python
acc_summary = tf.summary.scalar('accuracy',acc)                   #生成准确率标量图  
merge_summary = tf.summary.merge([acc_summary ,...(其他要显示的信息)])  #这里的[]不可省
```

如果要在tensorboard中画多个数据图，需定义多个tf.summary.FileWriter并重复上述过程。











​                



