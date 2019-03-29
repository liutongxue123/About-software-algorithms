# 关于Python 

## 一、Python语法







## 二、第三方模块的使用

### （1）opencv

opencv-python语法

**常用函数索引**

1、cv2.resize

cv2.resize(src,dsize,dst=None,fx=None,fy=None,interpolation=None)

参数：

scr:原图

dsize：输出图像尺寸

fx:沿水平轴的比例因子

fy:沿垂直轴的比例因子

interpolation：插值方法

代码实例：

```python
import cv2
import numpy as np
 
img = cv2.imread('McGrady.jpg')
 
#方法一：
# res = cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)  #比例因子：fx=2,fy=2
 
# 方法二：
##print(img.shape)         #输出为（450，600，3）
##print(img.shape[:2])   #输出为（450，600）
height,width = img.shape[:2]  #获取原图像的水平方向尺寸和垂直方向尺寸。
res = cv2.resize(img,(2*width,2*height),interpolation=cv2.INTER_CUBIC)   #dsize=（2*width,2*height）
 
while(1):
    cv2.imshow('res',res)
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0XFF ==27:
        break
cv2.destroyWindow()
```

2、cv2.imread()

读入图片，共两个参数，第一个参数为要读入的图片文件名，第二个参数为如何读取图片，包括cv2.IMREAD_COLOR：读入一副彩色图片；cv2.IMREAD_GRAYSCALE：以灰度模式读入图片；cv2.IMREAD_UNCHANGED：读入一幅图片，并包括其alpha通道。

3、cv2.imshow()

创建一个窗口显示图片，共两个参数，第一个参数表示窗口名字，可以创建多个窗口中，但是每个窗口不能重名；第二个参数是读入的图片。

4、cv2.waitKey()

键盘绑定函数，共一个参数，表示等待毫秒数，将等待特定的几毫秒，看键盘是否有输入，返回值为ASCII值。如果其参数为0，则表示无限期的等待键盘输入。

5、cv2.destroyAllWindows()

删除建立的全部窗口。

6、cv2.destroyWindows()

删除指定的窗口。

7、cv2.imwrite()

保存图片，共两个参数，第一个为保存文件名，第二个为读入图片。

### （2）PIL

最全的PythonPIL详解：https://baijiahao.baidu.com/s?id=1595108270577043146&wfr=spider&for=pc

Python图像处理PIL各模块详细介绍：https://blog.csdn.net/zhangziju/article/details/79123275

PIL：

Python Imaging Library，是Python中的图像处理标准库

在Debian/Ubuntu Linux下直接通过apt安装：

```
$ sudo apt-get install python-imaging
```

Pillow：

由于PIL仅支持到Python 2.7，加上年久失修，于是一群志愿者在PIL的基础上创建了兼容的版本，名字叫Pillow，支持最新Python 3.x，又加入了许多新特性，因此，我们可以直接安装使用Pillow。

如果安装了Anaconda，Pillow就已经可用了。否则，需要在命令行下通过pip安装：

```
$ pip install pillow
```

如果遇到`Permission denied`安装失败，请加上`sudo`重试。

**PIL基本概念：**

PIL中所涉及的基本概念有如下几个：

**通道（bands）、模式（mode）、尺寸（size）**

**坐标系统（coordinate system）、调色板（palette）**

**信息（info）和滤波器（filters）**

**1、通道**

​	每张图片都是由一个或者多个数据通道构成。PIL允许在单张图片中合成相同维数和深度的多个通道。以RGB图像为例，每张图片都是由三个数据通道构成，分别为R、G和B通道。而对于灰度图像，则只有一个通道。对于一张图片的通道数量和名称，可以通过**getbands()** 方法来获取。

​	getbands()方法是Image模块的方法，它会返回一个字符串元组（tuple）。该元组将包括每一个通道的名称。Python的元组与列表类似，不同之处在于元组的元素不能修改,元组使用小括号，列表使用方括号，元组创建很简单，只需要在括号中添加元素，并使用逗号隔开即可。

getbands()方法的使用如下：

![](D:\typora图片总\PIL1.jpg)

**2、模式**

图像的模式定义了图像的类型和像素的位宽。当前支持如下模式：

![](D:\typora图片总\PIL2.jpg)

mode属性的使用：

![](D:\typora图片总\PIL3.jpg)

**3、尺寸**

通过size属性可以获取图片的尺寸。这是一个二元组，包含水平和垂直方向上的像素数。

size属性的使用如下：

![](D:\typora图片总\PIL4.jpg)

**4、坐标系统**

PIL使用笛卡尔像素坐标系统，坐标(0，0)位于左上角。注意：**坐标值表示像素的角**；位于坐标（0，0）处的像素的中心实际上位于（0.5，0.5）。坐标经常用于二元组（x，y）。长方形则表示为四元组，前面是左上角坐标。例如：一个覆盖800x600的像素图像的长方形表示为（0，0，800，600）。

**5、调色板**

调色板模式 ("P")使用一个颜色调色板为每个像素定义具体的颜色值

**6、信息**

使用info属性可以为一张图片添加一些辅助信息。这个是字典对象。加载和保存图像文件时，多少信息需要处理取决于文件格式。

info属性的使用如下：

![](D:\typora图片总\PIL5.jpg)

**7、滤波器**

对于将多个输入像素映射为一个输出像素的几何操作，PIL提供了4个不同的采样滤波器：

注意：在当前的PIL版本中，ANTIALIAS滤波器是下采样时唯一正确的滤波器。**BILIEAR和BICUBIC滤波器使用固定的输入模板**，用于固定比例的几何变换和上采样是最好的。Image模块中的方法resize()和thumbnail()用到了滤波器。

**resize()方法**的定义为：resize(size, filter=None)=> image

使用方法：对参数filter不赋值的话，resize()方法默认使用NEAREST滤波器。如果要使用其他滤波器可以通过下面的方法来实现：

![](D:\typora图片总\PIL6.jpg)

**thumbnail ()方法**的定义为：im.thumbnail(size, filter=None)

使用方法：

![](D:\typora图片总\PIL7.jpg)

对参数filter不赋值的话，方法thumbnail()默认使用NEAREST滤波器。如果要使用其他滤波器可以通过下面的方法来实现：

![](D:\typora图片总\PIL8.jpg)



**PIL有如下模块：** 

Image模块、ImageChops模块、ImageCrackCode模块、ImageDraw模块、ImageEnhance模块、ImageFile模块、ImageFileIO模块、ImageFilter模块、ImageFont模块、ImageGrab模块、ImageOps模块、ImagePath模块、ImageSequence模块、ImageStat模块、ImageTk模块、ImageWin模块、PSDraw模块

**1、Image模块**

**Image模块是PIL中最重要的模块**，比如创建、打开、显示、保存图像等功能，合成、裁剪、滤波等功能，获取图像属性功能，如图像直方图、通道数等。

https://blog.csdn.net/zhangziju/article/details/79123275

**<1>open类**



**<2>Save类**



**<3>format类**



**<4>Mode类**



**<5>convert类**



**<6>Size类**



**<7>Palette类**



**<8>Info类**



**<9>new类**



**<10>Copy类**



**<11>Crop类**



**<12>Paste类**



**<13>Filter类**



**<14>Blend类**



**<15>Split类**



**<16>Composite类**



**<17>Eval类**



**<18>Merge类**



**<19>Draft类**



**<20>Getbands类**



**<21>Getbbox类**



**<22>Getdata类**



**<23>Getextrema类**



**<24>Getpixel类**



**<25>Histogram类**



**<26>Load类**



**<27>Putdata类**



**<28>Resize类**



**<29>Rotate类**



**<30>Seek类**



**<31>Tell类**



**<32>Thumbnail类**



**<33>Transform类**



**<34>Transpose类**













**2、ImageChops模块**



**3、ImageCrackCode模块**



**4、ImageDraw模块**



**5、ImageEnhance模块**



**6、ImageFile模块**



**7、ImageFileIO模块**



**8、ImageFilter模块**



**9、ImageFont模块**



**10、ImageGrab模块**



**11、ImageOps模块**



**12、ImagePath模块**



**13、ImageSequence模块**



**14、ImageStat模块**





**15、ImageTk模块**



**16、PSDraw模块**

















