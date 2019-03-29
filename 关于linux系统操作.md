# 关于linux系统操作

## 1、操作系统简介

​	如今的 IT 服务器领域是 Linux、UNIX、Windows 三分天下，Linux 系统可谓后起之秀，特别是“互联网热”以来，Linux 在服务器端的市场份额不断扩大，每年增长势头迅猛，开始对 Windows 和 UNIX 的地位构成严重威胁。下图为2016年初国内服务器端各个操作系统的市场份额：

![](D:\typora图片总\linux1.png)

​	可以看出来，Linux 占 80% 左右（包括 CentOS、Ubuntu 等），Windows 占 12.8%，Solaris 占 6.2%。在未来的服务器领域，Linux 是大势所趋。

​	Linux 在服务器上的应用非常广泛，可以用来搭建Web服务器、数据库服务器、负载均衡服务器（CDN）、邮件服务器、DNS服务器、反向代理服务器、VPN服务器、路由器等。用 Linux 作为服务器系统不但非常高效和稳定，还不用担心版权问题，不用付费。









## 1、关于linux挂载--mount

（1）、在windows系统中，一般c盘是主分区，后面的D盘、E盘等是逻辑分区。

![](D:\typora图片总\windows系统分区.png)

这些盘的概念，是windows赋予的，其实质就是一个个的分区。

当启动windows时，PC会自动检测电脑已连接的所有硬盘上能够识别的分区（NTFS、FAT32等），并自动为其分配盘符。这个分配盘符的过程，就是挂载（mount）的过程，简单来说，就是windows把第一分区关联到C这个盘符，把第二分区关联到D这个盘符。

（2）、在linux系统中，整个系统从根目录开始，按树形目录依次向下逐渐扩大，分类存放不同用途的文件。

第一个**/**表示根目录，后面的**/**表示路径分割符。

整个linux的系统结构里，有且只有一个root（根分区），不可能有第二个。

**注** ：实际上，任何一个分区都可以挂载为/，不过挂载根分区的目的是启动系统，若/下面没有linux内核及其所需的系统文件的话，将无法引导系统。

其他分区只能被继续挂载到/（根分区）下的某个目录里，比如/ mnt 或者/ media目录里。挂载好之后，当向这个目录读写数据时，其实是在向被挂载到该目录的另一个分区读写数据。多个分区在同一时间段只能被挂载到多个不同的目录，任何一个处于/根目录下的目录，都可以用来作为挂载到其他分区的平台。

**注：**“/”相当于c:\，而“/etc、/bin、/sbin、/lib”这些目录大概相当于c:\windows和c:\program files，“/home”相当于c:\Documents and Settings，而把第二分区挂载到“/mnt/partition2”的时候，这个目录就相当于d:\了



**挂载代码：**

cd /mnt                  （切换到/mnt目录）
sudo mkdir partition2      （新建一个名为partition2的空目录，你可以随意用其他名称）
**sudo mount /dev/sda5 partition2**   （如果你只有一个硬盘且第二分区是逻辑分区的话，这个命令就将挂载该分区到partition2）
cd partition2                 （切换到/mnt/partition2目录）
ls                                   （列出该目录的文件）

**反挂载：**

sudo umount /dev/sda5   （或者/mnt/partition2）

## 2、关于linux常用操作

#### （1）安装虚拟环境venv

```
$ virtualenv --system-site-packages -p python3 ./venv
$ source ./venv/bin/activate
$ pip install --upgrade pip
$ pip list
$ deactivate   #退出虚拟环境

在虚拟环境中安装tensorflow:
$ pip install --upgrade tensorflow
$ sudo apt-get install python3-tk
```



#### （2）安装jupyter notebook

unbuntu部署python2和python3共存的Jupyter Notebook

**1、安装python和python-pip**

```
sudo apt-get install python python3 python-pip python3-pip
sudo pip install --upgrade pip  #更新pip
sudo pip3 install --upgrade pip
```

**2、安装jupyter-notebook**

```
sudo pip install jupyter
sudo pip3 install jupyter
```

**3、配置可以同时使用python2和python3内核**

```
sudo ipython kernel install--user
sudo python3 -m ipykernel install--user
sudo pip2 install -U ipykernel
sudo python2 -m ipykernel install--user
```

**4、解决新建文件的错误：**

```
sudo chmod 777 ~/.local/share/jupyter
cd ~/.local/share/jupyter
ls
sudo chmod 777 runtime
cd runtime
ls
```

**5、添加到环境变量并运行**

**export PATH=$PATH:~/.local/bin**  #当前会话生效

**sudo jupyter-notebook 或者 ~/.local/bin/jupyter-notebook**  #运行，会自动打开web界面，可以同时运行python2，python3，ctrl+c结束

把export PATH=$PATH:~/.local/bin 添加到的最后一行 ~/.bashrc ，永久生效

**6、设置**

sudo jupyter-notebook password  #设置密码

生成配置文件：

sudo jupyter notebook --generate-config

sudo vim /root/.jupyter/jupyter_notebook_config.py

```
c.NotebookApp.ip = '*' #访问ip限制
c.NotebookApp.notebook_dir = '/home/knmax/Desktop/Python/jupyter-project'  #工作目录,路径不能出现中文
c.NotebookApp.open_browser = False #不自动打开浏览器
c.NotebookApp.port = 88 #运行监听的端口,默认是8888
```

启动时遇到的问题：

[blue@blue-Box](mailto:blue@blue-Box):~$ jupyter-notebook

jupyter-notebook: command not found

A:环境变量问题，参照第四步的配置`export PATH=$PATH:~/.local/bin`

可以使用这个命令启动：~/.local/bin/jupyter-notebook

#### （3）vnc操作

1、vncserver                           #开端口

2、vncserver -geometry        # 改分辨率+开端口

3、vncserver -list                    # 查看目前用户已开端口

4、vncserver -kill                    #端口号 删除端口号

#### （4）实验室服务器操作

**1、取得权限**

命令：su amax

密码：Amax1979

**2、使用图形化界面**

先使用vnc登陆服务器，再输入命令：nautilus





## 3、关于Shell编程

### 一、Shell简介

​	对于图形界面，用户点击某个图标就能启动某个程序；对于命令行，用户输入某个程序的名字（可以看做一个命令）就能启动某个程序。这两者的基本过程都是类似的，都需要查找程序在硬盘上的安装位置，然后将它们加载到内存运行。（关于程序的运行原理，可以参考《载入内存，让程序运行起来》http://c.biancheng.net/cpp/html/2838.html）。换句话说，图形界面和命令行要达到的目的是一样的，都是让用户控制计算机。然而，真正能够控制计算机硬件（CPU、内存、显示器等）的只有操作系统内核（Kernel），图形界面和命令行只是架设在用户和内核之间的一座桥梁。

​	由于安全、复杂、繁琐等原因，用户不能直接接触内核（也没有必要），需要另外再开发一个程序，让用户直接使用这个程序；该程序的作用就是接收用户的操作（点击图标、输入命令），并进行简单的处理，然后再传递给内核。如此一来，用户和内核之间就多了一层“代理”，这层“代理”既简化了用户的操作，也保护了内核。用户界面和命令行就是这个另外开发的程序，就是这层“代理”。在Linux下，这个命令行程序叫做 **Shell**。

​	Shell 除了能解释用户输入的命令，将它传递给内核，还可以：

- 调用其他程序，给其他程序传递数据或参数，并获取程序的处理结果；
- 在多个程序之间传递数据，把一个程序的输出作为另一个程序的输入；
- Shell 本身也可以被其他程序调用。

**由此可见，Shell 是将内核、程序和用户连接了起来。** 

![](D:\typora图片总\shell1.png)

​	Shell 本身支持的命令并不多，但是它可以调用其他的程序，每个程序就是一个命令，这使得 Shell 命令的数量可以无限扩展，其结果就是 Shell 的功能非常强大，完全能够胜任 Linux 的日常管理工作，如文本或字符串检索、文件的查找或创建、大规模软件的自动部署、更改系统设置、监控服务器性能、发送报警邮件、抓取网页内容、压缩文件等。

​	Shell 并不是简单的堆砌命令，我们还可以在 Shell 中编程，这和使用 C/C++、Java、Python 等常见的编程语言并没有什么两样。

​	Shell 虽然没有 C/C++、Java、Python 等强大，但也支持了基本的编程元素，例如：

- if...else 选择结构，switch...case 开关语句，for、while、until 循环；
- 变量、数组、字符串、注释、加减乘除、逻辑运算等概念；
- 函数，包括用户自定义的函数和内置函数（例如 printf、export、eval 等）。

​	站在这个角度讲，Shell 也是一种**编程语言**，它的编译器（解释器）是 Shell 这个程序。我们平时所说的 Shell，**有时候是指连接用户和内核的这个程序，有时候又是指 Shell 编程。**

**当shell是一种脚本语言：**

任何代码最终都要被“翻译”成二进制的形式才能在计算机中执行。

- 有的编程语言，如 C/C++、Pascal、Go语言、汇编等，必须在程序运行之前将所有代码都翻译成二进制形式，也就是生成可执行文件，用户拿到的是最终生成的可执行文件，看不到源码。

这个过程叫做**编译（Compile）**，这样的编程语言叫做**编译型语言**，完成编译过程的软件叫做**编译器（Compiler）**。

- 而有的编程语言，如 Shell、JavaScript、Python、PHP等，需要一边执行一边翻译，不会生成任何可执行文件，用户必须拿到源码才能运行程序。程序运行后会即时翻译，翻译完一部分执行一部分，不用等到所有代码都翻译完。

这个过程叫做**解释**，这样的编程语言叫做**解释型语言**或者**脚本语言（Script）**，完成解释过程的软件叫做**解释器**。



编译型语言的优点是执行速度快、对硬件要求低、保密性好，适合开发操作系统、大型应用程序、数据库等。

脚本语言的优点是使用灵活、部署容易、跨平台性好，非常适合Web开发以及小工具的制作。

Shell 就是一种脚本语言，我们编写完源码后不用编译，直接运行源码即可。  



能用于linux运维的脚本语言有**Shell**、**Python** 和 **Perl**

**（1）Perl 语言** 

Perl 比 Shell 强大很多，在 2010 年以前很流行，它的语法灵活、复杂，在实现不同的功能时可以用多种不同的方式，缺点是不易读，团队协作困难。

Perl 脚本已经成为历史了，现在的 Linux 运维人员几乎不需要了解 Perl 了，最多可以了解一下 Perl 的安装环境。

**（2）Python 语言** 

Python 是近几年非常流行的语言，它不但可以用于脚本程序开发，也可以实现 Web 程序开发（知乎、豆瓣、YouTube、Instagram 都是用 Python 开发），甚至还可以实现软件的开发（大名鼎鼎的 OpenStack、SaltStack 都是 Python 语言开发）、游戏开发、大数据开发、移动端开发。

现在越来越多的公司要求运维人员会 Python 自动化开发，Python 也成了运维人员必备的技能，每一个运维人员在熟悉了 Shell 之后，都应该再学习 Python 语言。

**（3）Shell**  

Shell 脚本的优势在于处理偏操作系统底层的业务，例如，Linux 内部的很多应用（有的是应用的一部分）都是使用 Shell 脚本开发的，因为有 1000 多个 Linux 系统命令为它作支撑，特别是 Linux 正则表达式以及三剑客 grep、awk、sed 等命令。

对于一些常见的系统脚本，使用 Shell 开发会更简单、更快速，例如，让软件一键自动化安装、优化，监控报警脚本，软件启动脚本，日志分析脚本等，虽然 Python 也能做到这些，但是考虑到掌握难度、开发效率、开发习惯等因素，它们可能就不如 Shell 脚本流行以及有优势了。对于一些常见的业务应用，使用 Shell 更符合 Linux 运维简单、易用、高效的三大原则。

Python 语言的优势在于开发复杂的运维软件、Web 页面的管理工具和Web业务的开发（例如 CMDB 自动化运维平台、跳板机、批量管理软件 SaltStack、云计算 OpenStack 软件）等。



### 二、常用的Shell

​	linux是一个开源的操作系统，由分布在世界各地的多个组织机构或个人共同开发完成，每个组织结构或个人负责一部分功能，最后组合在一起，就构成了今天的 Linux。例如：

- Linux 内核最初由芬兰黑客 Linus Torvalds 开发，后来他组建了团队，Linux 内核由这个团队维护。
- GNU 组织开发了很多核心软件和基础库，例如 GCC 编译器、C语言标准库、文本编辑器 Emacs、进程管理软件、Shell 以及 GNOME 桌面环境等。
- VIM 编辑器由荷兰人 Bram Moolenaar 开发。

​	Windows、Mac OS、Android 等操作系统不一样，它们都由一家公司开发，所有的核心软件和基础库都由一家公司做决定，容易形成统一的标准，一般不会开发多款功能类似的软件。

​	而 Linux 不一样，它是“万国牌”，由多个组织机构开发，不同的组织机构为了发展自己的 Linux 分支可能会开发出功能类似的软件，它们各有优缺点，用户可以自由选择。Shell 就是这样的一款软件，不同的组织机构开发了不同的 Shell，它们各有所长，有的占用资源少，有的支持高级编程功能，有的兼容性好，有的重视用户体验。

​	常见的 Shell 有 sh、bash、csh、tcsh、ash 等。

**1、sh**

sh 的全称是 Bourne shell，由 AT&T 公司的 Steve Bourne开发，为了纪念他，就用他的名字命名了。

sh 是 UNIX 上的标准 shell，很多 UNIX 版本都配有 sh。sh 是第一个流行的 Shell。  

**2、csh**

sh 之后另一个广为流传的 shell 是由柏克莱大学的 Bill Joy 设计的，这个 shell 的语法有点类似C语言，所以才得名为 C shell ，简称为 csh。

Bill Joy 是一个风云人物，他创立了 BSD 操作系统，开发了 vi 编辑器，还是 Sun 公司的创始人之一。  

BSD 是 UNIX 的一个重要分支，后人在此基础上发展出了很多现代的操作系统，最著名的有 FreeBSD、OpenBSD 和 NetBSD，就连 Mac OS X 在很大程度上也基于BSD。

**3、tcsh**

tcsh 是 csh 的增强版，加入了命令补全功能，提供了更加强大的语法支持。

**4、ash**

一个简单的轻量级的 Shell，占用资源少，适合运行于低内存环境，但是与下面讲到的 bash shell 完全兼容。

**5、bash** 

bash shell 是 Linux 的默认 shell，本教程也基于 bash 编写。

bash 由 GNU 组织开发，保持了对 sh shell 的兼容性，是各种 Linux 发行版默认配置的 shell。  

bash 兼容 sh 意味着，针对 sh 编写的 Shell 代码可以不加修改地在 bash 中运行。

尽管如此，bash 和 sh 还是有一些不同之处：

- 一方面，bash 扩展了一些命令和参数；
- 另一方面，bash 并不完全和 sh 兼容，它们有些行为并不一致，但在大多数企业运维的情况下区别不大，特殊场景可以使用 bash 代替 sh。

### **三、查看Shell：**

​	Shell 是一个程序，一般都是放在`/bin`或者`/user/bin`目录下，当前 Linux 系统可用的 Shell 都记录在`/etc/shells`文件中。`/etc/shells`是一个纯文本文件，你可以在图形界面下打开它，也可以使用 cat 命令查看它。

通过 cat 命令来查看当前 Linux 系统的可用 Shell：

```shell
$ cat /etc/shells
/bin/sh
/bin/bash
/sbin/nologin
/usr/bin/sh
/usr/bin/bash
/usr/sbin/nologin
/bin/tcsh
/bin/csh
```

在现代的 Linux 上，sh 已经被 bash 代替，`/bin/sh`往往是指向`/bin/bash`的符号链接。

如果你希望查看当前 Linux 的默认 Shell，那么可以输出 SHELL 环境变量：

```shell
$ echo $SHELL
/bin/bash
```

输出结果表明默认的 Shell 是 bash。

`echo`是一个 Shell 命令，用来输出变量的值，`SHELL`是 Linux 系统中的环境变量，它指明了当前使用的 Shell 程序的位置，也就是使用的哪个 Shell。

可以在当前Shell下运行子Shell，例如输入命令：**/bin/csh**，开始使用csh，想要退出csh，输入命令：**exit**

**bash的优势：** 

:one: 可以使用上下箭头，快速查看历史输入命令

:two:  输入部分字母后，双击Tab键，可以输出所有以此字母开头的命令

:three:  输入help，输出所有的命令



### 四、Shell提示符：

启动终端模拟包或者从 Linux 控制台登录后，便可以看到 Shell提示符。提示符是通往 Shell 的大门，是输入 Shell 命令的地方。

对于普通用户，Base shell 默认的提示符是美元符号`$`；对于超级用户（root 用户），Bash Shell 默认的提示符是井号`#`。该符号表示 Shell 等待输入命令。

不同的 Linux 发行版使用的提示符格式不同。例如在 CentOS 中，默认的提示符格式为：

```shell
[mozhiyan@localhost ~]$
```

这种格式包含了以下三个方面的信息：

- 启动 Shell 的用户名，也即 mozhiyan；
- 本地主机名称，也即 localhost；
- 当前目录，波浪号`~`是主目录的简写表示法。

Shell 通过PS1和PS2两个环境变量来控制**提示符**格式：

- PS1 控制最外层命令行的提示符格式。
- PS2 控制第二层命令行的提示符格式。

在 Shell 中初次输入命令，使用的是 PS1 指定的提示符格式；如果输入一个命令后还需要输入附加信息，Shell 就使用 PS2 指定的提示符格式。请看下面的例子：

```shell
[mozhiyan@localhost ~]$ echo "C语言中文网"
C语言中文网

[mozhiyan@localhost ~]$ echo "http://c.biancheng.net"
http://c.biancheng.net

[mozhiyan@localhost ~]$ echo "
> yan
> chang
> sheng
> "
yan
chang
sheng
[mozhiyan@localhost ~]$ 
```

**echo 是一个输出命令，可以用来输出数字、变量、字符串等**；本例中，我们使用 echo 来输出字符串。

字符串是一组由`" "`包围起来的字符序列，echo 将第一个`"`作为字符串的开端，将第二个`"`作为字符串的结尾。此处的字符串就可以看做 echo 命令的附加信息。

本例中，前两次使用 echo 命令时都是在后面紧跟字符串，一行之内输入了完整的附加信息。第三次使用 echo 时，将字符串分成多行，echo 遇到第一个`"`认为是不完整的附加信息，所以会继续等待用户输入，直到遇见第二个`"`。输入的附加信息就是第二层命令，所以使用`>`作为提示符。

要显示提示符的当前格式，可以使用 echo 输出 PS1 和 PS2：

```shell
[mozhiyan@localhost ~]$ echo $PS1
[\u@\h \W]\$
[mozhiyan@localhost ~]$ echo $PS2
>
[mozhiyan@localhost ~]$ 
```

Shell 使用以`\`为前导的特殊字符来表示命令提示符中包含的要素，这使得 PS1 和 PS2 的格式看起来可能有点奇怪。下表展示了可以在 PS1 和 PS2 中使用的特殊字符。

![](D:\typora图片总\shell2.png)

注意，所有的特殊字符均以反斜杠`\`开头，目的是与普通字符区分开来。您可以在命令提示符中使用以上任何特殊字符的组合。

我们可以通过修改 PS1 变量来修改提示符格式，例如：

```shell
[mozhiyan@localhost ~]$ PS1="[\t][\u]\$ "
[17:27:34][mozhiyan]$ 
```

新的 Shell 提示符现在可以显示当前的时间和用户名。不过这个新定义的 PS1 变量只在当前 Shell 会话期间有效，再次启动

 Shell 时将重新使用默认的提示符格式。

### 五、Bash Shell的安装与升级

#### （1）确定已有的Shell版本

如果安装的 Linux 是 RedHat、CentOS、Fedora、Ubuntu、Debian 等主流发行版，那么在你的系统中很可能已经预装了 Bash Shell，只需要确认一下是否确实已经安装以及预装的版本即可。具体的方法是：

```shell
# 确认系统中使用的 Shell 是 bash
$ echo $SHELL
/bin/bash

# 查看系统中 Bash Shell 的版本（方法一）
$ bash --version
GNU bash, version 3.2.25(1)-release (i686-redhat-linux-gnu)
Copyright (C) 2005 Free Software Foundation, Inc.

# 查看系统中 Bash Shell 的版本（方法二）
$ echo $BASH_VERSION
3.2.25(1)-release
```



#### （2）源码方式安装Bash

​	Linux 下安装软件的方式无非是 RPM 包安装、yum 安装、源码安装三种方式，可以任选一种方式。相对来说 RPM 包

安装和 yum 安装方式比较简单，若再考虑各种包的依赖关系，这两种方式中又属 yum 安装更为简单。

​	这里示范使用**源码安装Bash**的过程。

首先访问 <http://www.gnu.org/software/bash/bash.html> 页面，在 Downloads 中选择一个下载的链接，这里选择中国科技大学提供的FTP下载目录：<ftp://mirrors.ustc.edu.cn/gnu/bash/>。

当前很多生产环境的系统中使用的 bash 版本还是 3.2 版，读者可以根据实际需要选择具体的版本。写本教程时，最新的版本是 4.2 版本，所以这里使用这个版本来做示范。

**1、使用wget下载最新的 bash 源码包**

具体操作如下：

```shell
$ wget ftp://mirrors.ustc.edu.cn/gnu/bash/bash-4.2.tar.gz
--2013-04-11 19:37:41--  ftp://mirrors.ustc.edu.cn/gnu/bash/bash-4.2.tar.gz
           => &#x60;bash-4.2.tar.gz'
Resolving mirrors.ustc.edu.cn... 202.141.160.110, 2001:da8:d800:95::110
Connecting to mirrors.ustc.edu.cn|202.141.160.110|:21... connected.
Logging in as anonymous ... Logged in!
==> SYST ... done.    ==> PWD ... done.
==> TYPE I ... done.  ==> CWD /gnu/bash ... done.
==> SIZE bash-4.2.tar.gz ... 7009201
==> PASV ... done.    ==> RETR bash-4.2.tar.gz ... done.
Length: 7009201 (6.7M)
100%[==========================================>] 7,009,201   1.93M/s   in 3.5s  
2013-04-11 19:37:46 (1.89 MB/s) - &#x60;bash-4.2.tar.gz' saved [7009201]
```

**2、解压源码包**

解压源码包并进入生成的目录中：

```shell
# 解压后会在当前目录下生成一个bash-4.2目录
$ tar zxvf bash-4.2.tar.gz

#进入目录bash-4.2
$ cd bash-4.2
$
```

**3、准备配置（configure）**

最简单的配置方式是直接运行当前目录下的 configure，这会将 bash 安装到 /usr/local 目录中，不过编译安装软件时，好的习惯是使用`--prefix`参数指定安装目录。所以这里采用下面的配置方式。该条命令将会产生大量的输出，一开始会检查系统的编译环境以及相关的依赖软件。

最常见的错误可能是系统中没有安装 gcc 造成无法继续，如果是这个原因，使用 yum install gcc 命令进行安装。如果配置过程出现致命错误会立即退出，请读者注意输出内容中的 error 部分。

```shell
$ ./configure --prefix=/usr/local/bash4.2
checking build system type... i686-pc-linux-gnu
checking host system type... i686-pc-linux-gnu
Beginning configuration for bash-4.2-release for i686-pc-linux-gnu
checking for gcc... gcc
checking for C compiler default output file name... a.out
checking whether the C compiler works... Yes
......(略去内容)......

#如果大量的 checking 没问题，则配置环境检测通过。如果读者看到如下的输出内容，说明配置成功
configure: creating ./config.status
config.status: creating Makefile
config.status: creating builtins/Makefile
config.status: creating lib/readline/Makefile
config.status: creating lib/glob/Makefile
config.status: creating lib/intl/Makefile
config.status: creating lib/malloc/Makefile
config.status: creating lib/sh/Makefile
config.status: creating lib/termcap/Makefile
config.status: creating lib/tilde/Makefile
config.status: creating doc/Makefile
config.status: creating support/Makefile
config.status: creating po/Makefile.in
config.status: creating examples/loadables/Makefile
config.status: creating examples/loadables/perl/Makefile
config.status: creating config.h
config.status: executing default-1 commands
config.status: creating po/POTFILES
config.status: creating po/Makefile
config.status: executing default commands

#如果配置成功，会在当前目录中生成Makefile
$ ll Makefile
-rw-r--r-- 1 root root 77119 Apr 11 19:49 Makefile
```

**4、正式编译**

```shell
#编译过程会产生大量输出
$ make
rm -f mksyntax
gcc  -DPROGRAM='"bash"' -DCONF_HOSTTYPE='"i686"'
-DCONF_OSTYPE='"linux-gnu"' -DCONF_MACHTYPE='"i686-pc-linux-gnu"'
-DCONF_VENDOR='"pc"'
-DLOCALEDIR='"/usr/local/bash4.2/share/locale"'
-DPACKAGE='"bash"' -DSHELL -DHAVE_CONFIG_H   -I.  -I. -I./include
-I./lib   -g  -o mksyntax ./mksyntax.c
......(略去内容)......
```

**5、安装**

有时在安装前也可以进行测试，但是一般情况下这不是必需的。

```shell
#非必要步骤：测试安装
#[root@localhost bash-4.2]# make test
#安装
$ make install

#安装其实就是将make产生的文件复制到指定的目录中，在这里指定的目录就是之前我们用 --prefix 参数指定的/usr/local，可以在该目录中发现bash4.2目录
$ ls -ld /usr/local/bash4.2/
drwxr-xr-x 4 root root 4096 Apr 11 20:08 /usr/local/bash4.2/
```

到此为止，最新版本的 bash 就已经安装好了，确切地说是安装到了 /usr/local/bash4.2 中。

#### （3）使用新版本的Bash Shell

​	虽然最新版的 bash 已经安装到系统中，但是还需要经过一些设置才能使用。首先需要将最新的 bash 的路径写到

 /etc/shells 中，以向系统注册新 Shell 的路径。可以采取直接编辑 /etc/shells 文件的方式，或者采用如下更简单的方式：

```shell
$ echo "/usr/local/bash4.2/bin/bash" >> /etc/shells

然后使用命令 chsh（change shell 的简写）修改登录 Shell。
$ chsh
Changing shell for root.
New shell [/bin/bash]: /usr/local/bash4.2/bin/bash #输入要修改的shell
Shell changed. #显示成功修改了shell

#此处chsh并没有附加参数，所以默认是修改root的shell，如要改变其他用户的登录shell，可以在后面跟上用户名，使用这种方式给用户john更改shell
$ chsh john
```

chsh 命令做的工作就是修改了 /etc/passwd 文件中登录 Shell 的路径，所以如果明白了 chsh 的原理，实际上可以手工编辑 /etc/passwd 文件，将 root 用户的这行改成下面的样子（这又一次印证了 Linux 中一切皆文件的说法）：

```shell
$ cat /etc/passwd | grep bash4.2
root:x:0:0:root:/root:/usr/local/bash4.2/bin/bash
```

最后还需要重新登录以获得 Shell，登录后再次验证一下当前的 Shell 版本。

```shell
$ echo $BASH_VERSION
4.2.0(1)-release

#请注意，如果这时候你使用下面的命令可能会犯迷糊：为什么版本是3.2.25呢？不是已经是4.2了吗？
$ bash --version
GNU bash, version 3.2.25(1)-release (i686-redhat-linux-gnu)
Copyright (C) 2005 Free Software Foundation, Inc.

#通过使用 whereis bash 命令可了解当前运行的 bash 命令真实运行的是/bin/bash，也就是说现在是在版本为 4.2 的 bash 中运行了一个 3.2.25 版本的 bash 命令。如果要想每次运行 bash 的时候使用的是 4.2 的版本，需要修改 PATH 变量的值，读者可以自行完成这个任务
$ whereis bash
bash: /bin/bash /usr/local/bash4.2 /usr/share/man/man1/bash.1.gz
```





### 六、第一个Shell脚本

打开文本编辑器，新建一个文本文件，并命名为 test.sh。（扩展名`sh`代表 shell，扩展名并不影响脚本执行，见名知意就好，若用 php 写 shell 脚本，扩展名就用`php`好了。）

在 test.sh 中输入代码：（使用 **read** 命令从 stdin 获取用户输入的内容并赋值给 PERSON 变量，最后在 stdout 上输出）

```shell
#!/bin/bash
echo "Hello World !"  #这是一条语句

echo "What is your name?"
read PERSON
echo "Hello, $PERSON"
```

第 1 行的`#!`是一个约定的标记，它告诉系统这个脚本需要什么解释器来执行，即使用哪一种Shell；后面的`/bin/bash`就是指明了解释器的具体位置。

第 2 行的`#`及其后面的内容是注释。Shell 脚本中所有以`#`开头的都是注释（当然以`#!`开头的除外）。

第 5 行中表示从终端读取用户输入的数据，并赋值给 PERSON 变量。**read** 命令用来从标准输入文件（Standard Input，stdin，一般就是指终端）读取用户输入的数据。

第 6 行表示输出变量 PERSON 的内容。注意在变量名前边要加上`$`，否则变量名会作为字符串的一部分处理。

### 七、执行Shell脚本

执行Shell脚本通常有两种方法。

#### **（1）作为可执行程序** 

**简单示例：** 

Shell脚本也是一种解释执行的程序，可以在终端直接调用（需要使用 chmod 命令给 Shell 脚本加上执行权限），如下：

```shell
$ cd demo  #切换到 test.sh 所在的目录
$ chmod +x ./test.sh  #使脚本具有执行权限
$ ./test.sh  #执行脚本
```

第 2 行中，`chmod +x`表示给 test.sh 增加执行权限。

第 3 行中，`./`表示当前目录，整条命令的意思是执行当前目录下的 test.sh 脚本。如果不写`./`，Linux会到系统路径（由 PATH 环境变量指定）下查找 test.sh，而系统路径下显然不存在这个脚本，所以会执行失败。

**<1>使用点号 “ . ”**

点号用于执行某个脚本，甚至脚本没有可执行权限也可以运行。有时候在测试运行某个脚本时可能并不想为此修改脚本权限，这时候就可以使用`.`来运行脚本，非常方便。

编写下面的代码并保存为 test.sh：

```shell
#!/bin/bash
echo "http://c.biancheng.net/shell/"
```

如果没有运行权限的话，用`./`执行就会有报错，但是若在其前面使用点号来执行就不会报错，如下所示：

```shell
$ ./test.sh
bash: .test.sh: Permission denied
```

使用`.`增加 test.sh 的执行权限，就可以正常运行了：

```shell
$ . ./test.sh
http://c.biancheng.net/shell/
```



**<2>使用source命令** 

与点号类似，**source** 命令也可读取并在当前环境中执行脚本，同时还可返回脚本中最后一个命令的返回状态；如果没有返回值则返回 0，代表执行成功；如果未找到指定的脚本则返回 false。

```shell
$ source test.sh
http://c.biancheng.net/shell/
```



#### **（2）作为解释器参数**

这种运行方式是，直接运行解释器，其参数就是 shell 脚本的文件名，如：

```shell
$ /bin/bash test.sh
http://c.biancheng.net/shell/
```

这种方式运行的脚本，不需要在第一行指定解释器信息，写了也没用。

### 八、Shell变量： 

​	变量是任何一种编程语言都必不可少的组成部分，变量用来存放各种数据。脚本语言在定义变量时通常不需要指明类型，直接赋值就可以，Shell变量也遵循这个规则。

​	在 Bash shell 中，每一个变量的值都是字符串，无论你给变量赋值时有没有使用引号，值都会以字符串的形式存储。

这意味着，Bash shell 在默认情况下不会区分变量类型，即使你将整数和小数赋值给变量，它们也会被视为字符串，这一

点和大部分的编程语言不同。

​	当然，如果有必要，你也可以使用 declare 关键字显式定义变量的类型，但在一般情况下没有这个需求，Shell 开发者在编写代码时自行注意值的类型即可。

#### （1）定义变量

Shell 支持以下三种定义变量的方式：

```shell
variable=value
variable='value'
variable="value"
```

**注意：赋值号`=`的周围不能有空格，这和熟悉的大部分编程语言都不一样。** 

variable 是变量名，value 是赋给变量的值。如果 value 不包含任何空白符（例如空格、Tab 缩进等），那么可以不使用引

号；如果 value 包含了空白符，那么就必须使用引号包围起来。使用单引号和双引号也是有区别的。

**举例说明使用单引号和双引号的区别：**

```shell
#!/bin/bash
url="http://c.biancheng.net"
website1='C语言中文网：${url}'
website2="C语言中文网：${url}"
echo $website1
echo $website2
```

运行结果为：

C语言中文网：${url}
C语言中文网：http://c.biancheng.net

- 以单引号`' '`包围变量的值时，单引号里面是什么就输出什么，即使内容中有变量和命令（命令需要反引起来）也会把它们原样输出。这种方式比较适合定义显示纯字符串的情况，即不希望解析变量、命令等的场景。
- 以双引号`" "`包围变量的值时，输出时会先解析里面的变量和命令，而不是把双引号中的变量名和命令原样输出。这种方式比较适合字符串中附带有变量和命令并且想将其解析后再输出的变量定义。

**注：如果变量的内容是数字，那么可以不加引号；如果真的需要原样输出就加单引号；其他没有特别要求的字符串等最好都加上双引号，定义变量时加双引号是最常见的使用场景。**

**Shell变量的命名规范：**

- 变量名由数字、字母、下划线组成；
- 必须以字母或者下划线开头；
- 不能使用 Shell 里的关键字（通过 help 命令可以查看保留关键字）。

**变量定义举例：**

```shell
url=http://c.biancheng.net/shell/
echo $url
name='C语言中文网'
echo $name
author="严长生"
echo $author
```

#### （2）使用变量

使用一个定义过的变量，只要在变量名前面加美元符号`$`即可，如：

```shell
author="严长生"
echo $author
echo ${author}
```

变量名外面的花括号`{ }`是可选的，加不加都行，加花括号是为了帮助解释器识别变量的边界，比如下面这种情况：

```shell
skill="Java"
echo "I am good at ${skill}Script"
```

如果不给 skill 变量加花括号，写成`echo "I am good at $skillScript"`，解释器就会把 $skillScript 当成一个变量（其值为空），代码执行结果就不是我们期望的样子了。

**注：建议给所有变量加上花括号`{ }`**

#### （3）修改变量的值

已定义的变量，可以被重新赋值，如：

```shell
url="http://c.biancheng.net"
echo ${url}
url="http://c.biancheng.net/shell/"
echo ${url}
```

第二次对变量赋值时不能在变量名前加`$`，只有在使用变量时才能加`$`。

#### （4）将命令的结果赋值给变量

Shell 也支持将命令的执行结果赋值给变量，常见的有以下两种方式：

```shell
variable=`command`
variable=$(command)
```

第一种方式把命令用反引号（位于 Esc 键的下方）包围起来，反引号和单引号非常相似，容易产生混淆，所以不推荐使用这种方式；第二种方式把命令用`$()`包围起来，区分更加明显，所以推荐使用这种方式。

示例：

在 code 目录中创建了一个名为 log.txt 的文本文件，用来记录我的日常工作。下面的代码中，使用 cat 命令将 log.txt 的内容读取出来，并赋值给一个变量，然后使用 echo 命令输出。

```shell
$ cd code
$ log=$(cat log.txt)            //方法一
$ echo $log
[2018-09-10 06:53:22] 严长生正在编写Shell教程

$ log=`cat log.txt`              //方法二
$ echo $log
[2018-09-10 06:53:22] 严长生正在编写Shell教程
$ 
```

#### （5）只读变量

使用 **readonly** 命令可以将变量定义为只读变量，只读变量的值不能被改变。

下面的例子尝试更改只读变量，结果报错：

```shell
#!/bin/bash
myUrl="http://see.xidian.edu.cn/cpp/shell/"
readonly myUrl
myUrl="http://see.xidian.edu.cn/cpp/danpianji/"
```

运行结果如下：

```shell
/bin/sh: NAME: This variable is read only.
```

#### （7）删除变量

使用 **unset** 命令可以删除变量。语法为：

```shell
unset variable_name
```

变量被删除后不能再次使用；unset 命令不能删除只读变量。

举例：

```shell
#!/bin/sh

myUrl="http://c.biancheng.net/shell/"
unset myUrl
echo $myUrl
```

上面的脚本没有任何输出。

#### （8）变量的作用域

​	Shell变量的**作用域（Scope）**，就是 Shell 变量的有效范围（可以使用的范围）。

​	在不同的作用域中，同名的变量不会相互干涉，就好像 A 班有个叫小明的同学，B 班也有个叫小明的同学，虽然他们都叫小明（对应于变量名），但是由于所在的班级（对应于作用域）不同，所以不会造成混乱。但是如果同一个班级中有两个叫小明的同学，就必须用类似于“大小明”、“小小明”这样的命名来区分他们。

Shell 变量的作用域可以分为三种：

- 有的变量可以在当前 Shell 会话中使用，这叫做**全局变量（global variable）**；
- 有的变量只能在函数内部使用，这叫做**局部变量（local variable）**；
- 而有的变量还可以在其它 Shell 中使用，这叫做**环境变量（environment variable）**。

##### 1、全局变量

所谓全局变量，就是指变量在当前的整个 Shell 会话中都有效。每个 Shell 会话都有自己的作用域，彼此之间互不影响。**在 Shell 中定义的变量，默认就是全局变量。**

想要实际演示全局变量在不同 Shell 会话中的互不相关性，可在图形界面下同时打开两个 Shell，或使用两个终端远程连接到服务器（SSH）。

首先打开一个 Shell 窗口，定义一个变量 a 并赋值为 1，然后打印，这时在同一个 Shell 窗口中是可正确打印变量 a 的值的。然后再打开一个新的 Shell 窗口，同样打印变量 a 的值，但结果却为空，如图所示：

![](D:\typora图片总\shell3.png)

​	这说明全局变量 a 仅仅在定义它的第一个 Shell 中有效，对其它 Shell 没有影响。

​	全局变量的作用范围是当前的 Shell 会话，而不是当前的Shell 脚本文件，它们是不同的概念。打开一个 Shell 窗口就创建了一个 Shell 会话，打开多个 Shell 窗口就创建了多个 Shell 会话，每个 Shell 会话都是独立的进程，拥有不同的进程 ID。在一个 Shell 会话中，可以执行多个 Shell 脚本文件，此时全局变量在这些脚本文件中都有效。

例如，现在有两个 Shell 脚本文件，分别是 a.sh 和 b.sh。

a.sh 的代码如下：

```shell
#!/bin/bash
echo $a
b=200
```

b.sh 的代码如下：

```shell
#!/bin/bash
echo $b
```

打开一个 Shell 窗口，输入以下命令：

```shell
$ a=99
$ . ./a.sh
99
$ . b.sh
200
$
```

从输出结果可以发现，在 Shell 会话中以命令行的形式定义的变量 a，在 a.sh 中有效；在 a.sh 中定义的变量 b，在 b.sh 中也有效。

##### 2、局部变量

Shell 也支持自定义函数，但是 Shell 函数和 C/C++、Java 等其他编程语言函数的一个不同点就是：**在 Shell 函数中定义的变量默认也是全局变量**，它和在函数外部定义变量拥有一样的效果。请看下面的代码：

```shell
#!/bin/bash
#定义函数
function func(){
    a=99
}
#调用函数
func
#输出函数内部的变量
echo $a
```

输出结果：
99

a 是在函数内部定义的，但是在函数外部也可以得到它的值，证明它的作用域是全局的，而不是仅限于函数内部。

要想变量的作用域仅限于函数内部，那么可以在定义时加上`local`命令，此时该变量就成了局部变量。请看下面的代码：

```shell
#!/bin/bash
#定义函数
function func(){
    local a=99
}
#调用函数
func
#输出函数内部的变量
echo $a
```

输出结果为空，表明变量 a 在函数外部无效，是一个局部变量。

Shell 变量的这个特性和 JavaScript 中的变量是类似的。在 JavaScript 函数内部定义的变量，默认也是全局变量，只有加

上`var`关键字，它才会变成局部变量。

##### 3、环境变量

​	全局变量只在当前 Shell 会话中有效，如果使用`export`命令将它导出，那么它就在所有的子 Shell 中也有效了，这称为“环境变量”。

​	环境变量被创建时所处的 Shell 被称为父 Shell，如果在父 Shell 中再创建一个 Shell，则该 Shell 被称作子 Shell。当子 Shell 产生时，它会继承父 Shell 的环境变量为自己所用，所以说环境变量可从父 Shell 传给子 Shell。不难理解，环境变量还可以传递给孙 Shell。

**注：环境变量只能向下传递而不能向上传递，即“传子不传父”**

在一个 Shell 中创建子 Shell 最简单的方式是运行 bash 命令，如下图所示：

![](D:\typora图片总\shell4.png)

通过`exit`命令可以一层一层地退出 Shell。

下面具体演示环境变量的使用：

```shell
$ a=22      #定义一个全局变量
$ echo $a    #在当前Shell中输出a，成功
22
$ bash    #进入子Shell
$ echo $a    #在子Shell中输出a，失败

$ exit    #退出子Shell
exit
$ export a    #将a导出为环境变量
$ bash    #重新进入子Shell
$ echo $a    #在子Shell中再次输出a，成功
22
$ exit    #退出子Shell
exit
$ exit    #退出父Shell，结束整个Shell会话
```

可以发现，默认情况下，a 在子 Shell 中是无效的；使用 export 将 a 导出为环境变量后，在子 Shell 中就可以使用了。

`export a`这种形式是在定义变量 a 以后再将它导出为环境变量，如果想在定义的同时导出为环境变量，可以写作

`export a=22`。

**注：** 

1、上面一直强调的是环境变量在子 Shell 中有效，并没有说它在所有的 Shell 种有效；如果通过终端创建一个新的 Shell，那么它就不是当前 Shell 的子 Shell，环境变量对这个 Shell 就是无效的。

2、此外，通过`export`命令导出的环境变量是临时的，关闭 Shell 会话后它就销毁了。所以，这种环境变量也只是在局部范围内起作用，并不影响所有 Shell。如果想让环境变量在所有 Shell 中都有效，并且能够永久保存，在关闭 Shell 后也不丢失，那么就需要把环境变量写入启动文件。

### 九、Shell位置参数

​	运行Shell 脚本文件时我们可以给它传递一些参数，这些参数在脚本文件**内部**可以使用`$n`的形式来接收，例如，\$1 表

示第一个参数，$2 表示第二个参数，依次类推。

​	同样，在调用函数时也可以传递参数。Shell 函数参数的传递和其它编程语言不同，没有所谓的形参和实参，在定义函

数时也不用指明参数的名字和数目。换句话说，定义 Shell 函数时不能带参数，但是在调用函数时却可以传递参数，这些传

递进来的参数，在函数内部就也使用`$n`的形式接收，例如，\$1 表示第一个参数，​$2 表示第二个参数，依次类推。

​	这种通过`$n`的形式来接收的参数，在 Shell 中称为位置参数。

​	上面提到过，变量的名字必须以字母或者下划线开头，不能以数字开头；但是位置参数却是数字，这和变量的命

名规则是相悖的，所以我们将它们视为“特殊变量”。

​	除了 \$n，Shell 中还有 \$#、\$*、\$@、\$?、$$ 几个特殊参数

#### （1）给脚本文件传递位置参数

请编写下面的代码，并命名为 test.sh：

```shell
#!/bin/bash

echo "Language: $1"
echo "URL: $2"
```

运行 test.sh，并附带参数：

```
$ . ./a.sh Shell http://c.biancheng.net/shell/
```

运行结果：
Language: Shell
URL: http://c.biancheng.net/shell/

其中`Shell`是第一个位置参数，`http://c.biancheng.net/shell/`是第二个位置参数。

#### （2）给函数传递位置参数

请编写下面的代码，并命名为 test.sh：

```shell
#!/bin/bash
#定义函数
function func(){
    echo "Language: $1"
    echo "URL: $2"
}
#调用函数
func C++ http://c.biancheng.net/cplus/
```

运行 test.sh：

```
$ . ./a.sh
```

运行结果：
Language: C++
URL: http://c.biancheng.net/cplus/

**注意：**

如果参数个数太多，达到或者超过了 10 个，那么就得用`${n}`的形式来接收了，例如 \${10}、​${23}。`{ }`的作用是为了帮助解释器识别参数的边界，这跟使用变量时加`{ }`是一样的效果。

### 十、Shell特殊变量：

**Shell \$#、​\$*、\$@、\$?、$$**

| 变量      | 含义                                                         |
| --------- | ------------------------------------------------------------ |
| $0        | 当前脚本的文件名。                                           |
| $n（n≥1） | 传递给脚本或函数的参数。n 是一个数字，表示第几个参数。例如，第一个参数是 $1，第二个参数是 $2。 |
| $#        | 传递给脚本或函数的参数个数。                                 |
| $*        | 传递给脚本或函数的所有参数。                                 |
| $@        | 传递给脚本或函数的所有参数。当被双引号`" "`包含时，$@ 与 $* 稍有不同 |
| $?        | 上个命令的退出状态，或函数的返回值                           |
| $$        | 当前 Shell 进程 ID。对于 Shell 脚本，就是这些脚本所在的进程 ID。 |

**实例演示：**

##### **（1）给脚本文件传递参数**

编写下面的代码，并保存为 test.sh：

```shell
#!/bin/bash

echo "Process ID: $$"
echo "File Name: $0"
echo "First Parameter : $1"
echo "Second Parameter : $2"
echo "All parameters 1: $@"
echo "All parameters 2: $*"
echo "Total: $#"
```

运行 test.sh，并附带参数：

```shell
$ chmod +x ./a.sh
$ ./test.sh Shell Linux
```

运行结果为：
Process ID: 2788
File Name: ./test.sh
First Parameter : Shell
Second Parameter : Linux
All parameters 1: Shell Linux
All parameters 2: Shell Linux
Total: 2

##### （2）给函数传递参数

编写下面的代码，并保存为 test.sh：

```shell
#!/bin/bash
#定义函数
function func(){
    echo "Language: $1"
    echo "URL: $2"
    echo "First Parameter : $1"
    echo "Second Parameter : $2"
    echo "All parameters 1: $@"
    echo "All parameters 2: $*"
    echo "Total: $#"
}

#调用函数
func Java http://c.biancheng.net/java/
```

运行结果为：
Language: Java
URL: http://c.biancheng.net/java/
First Parameter : Java
Second Parameter : http://c.biancheng.net/java/
All parameters 1: Java http://c.biancheng.net/java/
All parameters 2: Java http://c.biancheng.net/java/
Total: 2















