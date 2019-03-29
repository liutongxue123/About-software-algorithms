# 关于C++

## 1、main：处理，命令行选项

假设main函数位于可执行文件prog之内，可以向程序传递下面的命令来执行程序：

prog -d -o ofile data0

这些命令通过两个（可选的）形参传递给main函数。

 int main ( int argc , char *argv[ ] )  {........}

第二个形参argv是一个数组，它的元素是指向C风格字符串的指针；

第一个形参argc表示数组中字符串的数量。

因为第二个形参是数组，所以main函数也可以定义成：

 int main ( int argc , char **argv[ ] )  {........}

其中argv指向char *

当实参传给main函数之后，**argv的第一个元素指向程序的名字或者一个空字符串，接下来的元素依次传递命令行传递的实参。最后一个指针之后的元素值保证为0。** 

以上面提供的命令行为例，argc应该等于5，argc应该包含如下的c风格字符串：

```c
argv [0] = "prog"; //或者argv[0]也可以指向一个空字符串
argv [1] = "-d";
argv [2] = "-o";
argv [3] = "ofile";
argv [4] = "data0";
argv [5] = "0";
```

