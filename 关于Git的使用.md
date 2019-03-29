---
typora-root-url: D:\typora图片总
---

### 0、命令合集

查看仓库状态：git status

查看文件修改内容：git diff

查看修改历史记录：git log

将文件添加到仓库：git add readme.txt

将文件提交到仓库：git commit -m "wrote a readme file"

撤销修改返回上一步：git checkout -- readme.txt

撤销添加到暂存区的修改：git reset HEAD readme.txt

把本地库的所有内容推送到远程库上：git push -u origin master

从远程库克隆：git clone git@github.com:liutongxue123/gitskills.git

查看分支：git branch

创建分支：git branch name

切换分支：git checkout name

创建+切换分支：git checkout -b name

合并某分支到当前分支：git merge name

删除分支：git branch -d name

新建标签：git tag name

指定标签信息：git tag -a tagname -m "blablabla..."

用PGP签名标签：git tag -s tagname -m "blablabla..."

查看所有标签：git tag

推送一个本地标签 ：git push origin tagname

推送全部未推送过的本地标签：git push origin --tags

删除一个本地标签：git tag -d tagname

删除一个远程标签：git push origin :refs/tags/tagname



### 1、Git简介

Git是一个分布式版本控制系统。版本控制系统是自动记录每次文件的改动。

**集中式版本控制系统和分布式版本控制系统的区别。**

 	集中式版本控制系统，版本库是几种存放在中央服务器的，而干活的时候，用的都是自己的电脑，所以要先从中央服务器取得最新的版本，然后开始干活，干完活了，再把自己的活推送给中央服务器。中央服务器就好比是一个图书馆，你要改一本书，必须先从图书馆借出来，然后回到家自己改，改完后再放回图书馆。其最大的缺点是必须联网才能工作。

![](/集中式版本控制系统.jpg)

​	分布式版本控制系统根本没有“中央服务器”，每个人的电脑上都有一个完整的版本库，这 样你工作的时候，就不需要联网了，因为版本库就在你自己的电脑上。

​	既然每个人电脑上都有一个完整的版本库，那多个人如何协作呢？比方说你在自己电脑上改 了文件A，你的同事也在他的电脑上改了文件A，这时，你们俩之间只需把各自的修改推送给对方，就可以互相看到对方的修改。

​	和集中式版本控制系统相比，分布式版本控制系统的安全性要高很多，因为每个人电脑里都有完整的版本库，某一个人的电脑坏掉了不要紧，随便从其他人那里复制一个就可以了。而集中式版本控制系统的中央服务器要是出了问题，所有人都没法干活了。在实际使用分布式版本控制系统的时候，其实很少在两人之间的电脑上推送版本库的修改，因为可能你们俩不在一个局域网内，两台电脑互相访问不了，也可能今天你的同事病了，他的电脑压根没有开机。因此，分布式版本控制系统通常也有一台充当“中央服务器”的电脑，但这个服务器的作用仅仅是用来方便“交换” 大家的修改，没有它大家也一样干活，只是交换修改不方便而已。

![](/分布式版本控制系统图.jpg)

### 2、Git的安装

**a.在linux上安装Git**

如果你碰巧用Debian或Ubuntu Linux，通过一条 “**sudo apt-get install git**” 就可以直接完成Git的安装，非常简单。

如果是其他Linux版本，可以直接通过源码安装。先从Git官网下载源码，然后解压，依次输入：./config，make，sudo make install 这几个命令安装就好了。

**b.在Mac OS X 上安装Git**

有两种安装方法：

一是安装homebrew，然后通过homebrew安装Git，具体方法请参考homebrew的文档：http://brew.sh/

二是直接从AppStore安装Xcode，Xcode集成了Git，不过默认没有安装，你需要运行 Xcode，选择菜单“Xcode”->“Preferences”，在弹出窗口中找到“Downloads”，选择“Command Line Tools”，点“Install”就可以完成安装了。

Xcode是Apple官方IDE，功能非常强大，是开发Mac和iOS App的必选装备，而且是免费的。

**c.在Windows上安装Git**

​	Windows下要使用很多Linux/Unix的工具时，需要Cygwin这样的模拟环境，Git也一样。Cygwin的安装和配置都比较复杂， 就不建议你折腾了。不过，有高人已经把模拟环境和Git都打包好了，名叫msysgit，只需要下载一个单独的exe安装程序，其他什么也不用装，绝对好用。

​	msysgit是Windows版的Git，下载地址为http://msysgit.github.io/，然后按照默认安装即可。

​	安装完成后，在开始菜单里找到 “Git” - > “Git Bash”，蹦出一个类似命令行窗口的东西，就说明Git安装成功！

安装完成后，还需要最后一步设置，在命令行窗口中输入如下命令：

 **git config --global user.name "Your Name"**
 **git config --global user.email "email@example.com"**

因为Git是分布式版本控制系统，所以，每个机器都必须自报家门：你的名字和Email地址。



### 3、版本库（仓库）的使用

**a.创建一个版本库**

什么是版本库呢？版本库又名仓库，英文名repository，你可以简单理解成一个目录，这个目录里面的所有文件都可以被Git管理起来，每个文件的修改、删除，Git都能跟踪，以便任何时刻都可以追踪历史，或者在将来某个时刻可以“还原”。

- 在linux中，先创建一个新的目录来作为仓库。使用命令：**mkdir learngit**（假设目录名称为learngit）

接下来用命令：**cd learngit** , 此时可以用**pwd**查看目录所在的路径 

然后在仓库目录下，通过**git init**命令把这个目录变成Git可以管理的仓 库。

执行完之后，会出现

**$ git init
Initialized empty Git repository in /Users/michael/learngit/.git/**

这样就把仓库建好了，而且告诉你是一个空的仓库（empty Git repository），此时可以发现当前目录下多了一个.git的目录，这个目录是Git来跟踪管理版本库的，没事千万不要手动修改这个目录里面的文件，不然改乱了，就把Git仓库给破坏了。

- 在Windows中，直接打开Git Bash中执行是上述步骤即可。

**b.把文件添加到版本库**

​	首先这里再明确一下，所有的版本控制系统，其实只能跟踪文本文件的改动，比如TXT文件，网页，所有的程序代码等等，Git也不例外。版本控制系统 可以告诉你每次的改动，比如在第5行加了一个单词“Linux”，在第8行删了一个单词“Windows”。而图片、视频这些二进制文件，虽然也能由版本控制系统管理，但没法跟踪文件的变化，只能把二进制文件每次改动串起来，也就是只知道图片从100KB改成了120KB，但到底改了啥，版本控制系统不知道，也没法知道。

​	不幸的是，Microsoft的Word格式是二进制格式，因此，版本控制系统是没法跟踪Word文件的改动的，前面我们举的例子只是为了演示，如果要真正使用版本控制系统，就要以纯文本方式编写文件。因为文本是有编码的，比如中文有常用的**GBK**编码，日文有**Shift_JIS**编码，如果没有历史遗留问题，强烈建议使用标准的**UTF-8**编码，所有语言使用同一种编码，既没有冲突，又被所有平台所支持。

**注：**当使用Windows时，千万不要使用自带的记事本txt编辑器，容易出现问题。可以使用Notepad++，注意把Notepad++的默认编码设置为UTF-8 without BOM即可。

**把文件放到Git仓库的步骤：**

1、先将要存放的文件先放到Git的文件目录下，其下面的子目录也行。

2、用git add命令将文件添加到仓库，如：**git add readme.txt**，执行完之后，没显示就是没错。

3、用git commit命令将文件提交到仓库，如：**git commit -m "wrote a readme file"**

​     -m后面输入的是本次提交的说明，最好有。

​     执行完之后的结果一般是：

​    [master (root-commit) cb926e7] wrote a readme file
   1 file changed, 2 insertions(+)
   create mode 100644 readme.txt 

**注：可以执行几次add添加文件，然后执行commit一次性提交多个文件到仓库。**

如：

\$ git add file1.txt
​\$ git add file2.txt
​\$ git add file3.txt
$ git commit -m "add 3 files."

### 4、本地仓库时光机操作

#### **（1）基本操作**

a. 使用**git status** 掌握仓库的状态。

b. 使用**git diff **命令查看文件的修改内容。

c. 确认修改内容完毕后，使用**git add** 和**git commit** 将修改后的文件提交到仓库。

d. 使用**git log**查看修改的历史记录。 如果加上参数--pretty=oneline，可以将记录简化。

​    看到的一大串类似“ 3628164...882e1e0”的是commit id（版本号） 

#### **（2）版本回退**

​	首先，Git必须知道当前版本是哪个版本，在Git中，用HEAD表示当前版本，也就是最新的提交“ 3628164...882e1e0”（注意我的提交ID和你的肯定不一样），上一个版本就是HEAD^，上上一个版本就是HEAD^^，当然往上100 个版本写100个^比较容易数不过来，所以写成HEAD~100。

a.  使用git reset命令回退到上一个版本，具体为：**git reset --hard HEAD^**

​    执行后，会显示如：HEAD is now at ea34578 add distributed

​    此时，可以用命令**cat**，查看此时文件的内容。

b.  返回到原来最新的版本。

​    当返回到上一版本后，再用**git log**命令查看，已经找不到最新版本的记录了。

   1、此时，想要返回原来最新的版本，需要保证命令窗口还没关闭，从上面找到新版本的commit id，然后执行命令，如：**git reset --hard 3628164**，版本号不需要全写，一般写前几位即可。

   2、如果此时已经将命令窗口关闭了，可以使用**git reflog**查看命令目录，这样就可以找到新版本的版本号。

#### （3）工作区和暂存区

**工作区：**就是电脑里能够实际看到的目录。

**版本库：**工作区有一个隐藏目录“.git”，这个不算工作区，而是Git的版本库。
               Git的版本库里存了很多东西，其中最重要的就是称为stage（或者叫index）的暂存区，还有Git为我们自  动创建的第一个分支master，以及指向master的一个指针叫HEAD。

![](/工作区和暂存区.jpg)

前面讲了我们把文件往Git版本库里添加的时候，是分两步执行的：

第一步是用“git add”把文件添加进去，实际上就是把文件修改添加到暂存区；

第二步是用“git commit”提交更改，实际上就是把暂存区的所有内容提交到当前分支。

如图所示：

![](/暂存区操作1.jpg)

![](/暂存区操作2.jpg)

#### （4）撤销修改

 a.   使用命令：**git checkout -- readme.txt**

命令git checkout -- readme.txt意思就是，把readme.txt文件在工作区的修改全部撤销，这里有两种情况：

一种是readme.txt自修改后还没有被放到暂存区，现在，撤销修改就回到和版本库一模一样的状态；

一种是readme.txt已经添加到暂存区后，又作了修改，现在，撤销修改就回到添加到暂存区后的状态。

总之，就是让这个文件回到最近一次git commit或git add时的状态。

 b.   使用命令：**git reset HEAD readme.txt**

适用的情况是：已经将修改add到了暂存区，还没有commit。

用此命令可以把暂存区的修改撤销掉（unstage），重新放回工作区。

#### （5）删除文件

a.首先，可以直接在文件管理器中删除文件或者使用**rm**命令删除。

这个时候，Git知道你删除了文件，因此，工作区和版本库就不一致了，使用git status命令会立
刻告诉你哪些文件被删除了。

b.接下来有两种情况：

1、从版本库中删除该文件，可以用命令**git rm**删掉，并且commit，如：

**git rm test.txt**
rm 'test.txt'
**git commit -m "remove test.txt"**
[master d17efd8] remove test.txt
1 file changed, 1 deletion(-)
delete mode 100644 test.txt

2、如果删除错了，因为此时版本库里还有，可以使用返回上一步命令：**git checkout -- test.txt**

### 5、远程仓库

找一台电脑充当服务器的角色，每天24小时开机，其他每个人都从这个“服务器”仓库克隆一份到自己的电脑上，并且各自把各自的提交推送到服务器仓库里，也从服务器仓库中拉取别人的提交。

GitHub就是提供Git仓库托管服务的。

由于本地Git仓库和GitHub仓库之间的传输是通过SSH加密的，所以，需要一点设置，步骤如下：

#### （1） 创建SSH Key

​	在用户主目录下，看看有没有.ssh目录，如果有，再看看这个目录下有没有id_rsa和id_rsa.pub这两个文件，如果已经有了，可直接 跳到下一步。如果没有，打开Shell（Windows下打开Git Bash），创建SSH Key：
**$ ssh-keygen -t rsa -C "youremail@example.com"**

需要把邮件地址换成你自己的邮件地址，然后一路回车，使用默认值即可，由于这个Key也不是用于军事目的，所

以也无需设置密码。如果一切顺利的话，可以在用户主目录里找到.ssh目录，里面有id_rsa和id_rsa.pub两个文

件，这两个就是SSH Key的秘钥对，id_rsa是私钥，不能泄露出去，id_rsa.pub是公钥，可以放心地告诉任何人。

#### （2） 在GitHub中设置

登陆GitHub，打开“Account settings”，“SSH Keys”页面：

然后，点“Add SSH Key”，填上任意Title，在Key文本框里粘贴id_rsa.pub文件的内容，然后点“Add Key”，就可以看到已经添加的Key。

GitHub通过此SSH Keys识别是否是本人推送的文件。

如果有多台电脑，可以在GitHub设置中添加多个SSH Keys。

**注：GitHub上免费托管的Git仓库，是公开的。**

#### （3）添加远程仓库

首先，登陆GitHub，然后，在右上角找到“New repository”按钮，来创建一个新的仓库，

在Repository name填入自己本地Git仓库的名称（现在为git-repository），其他保持默认设置，点击“Create 

repository”按钮，就成功地创建了一个新的Git仓库。成功后显示如下：

![](/github添加本地库.png)

然后，根据GitHub的提示，在本地仓库运行一下命令：

**git remote add origin https://github.com/liutongxue123/git-repository.git**

添加后，远echo程库的名字就是origin，这是Git默认的叫法，也可以改成别的，但是origin这个名字一看就知道是远程库。

接下来，就可以把本地库的所有内容推送到远程库上：

**git push -u origin master**

把本地库的内容推送到远程，用git push命令，实际上是把当前分支master推送到远程。

​	由于远程库是空的，我们第一次推送master分支时，加上了-u参数，Git不但会把本地的master分支内容推送到远程新的master分支，还会把本地的master分支和远程master分支关联起来，在以后的推送或者拉取时就可以简化命令。

执行此命令后，会按照提示，填入GitHub的账户名和密码。执行完之后。显示如下：

![](/推送本地Git到远程GitHub.png)

推送成功后，可以立刻在GitHub页面中看到远程库的内容已经和本地一模一样：

![](/将本地Git推送到远程GitHub仓库.png)

在这之后，只要本地作了提交，就可以通过命令：

**git push origin master**

把本地master分支的最新修改推送至GitHub，现在，已经拥有了真正的分布式版本库。

#### （4）从远程库克隆

以上的教程是讲述先建立本地仓库，再建立远程仓库，然后将本地仓库和远程仓库相关联。

接下来的教程是，先建立远程仓库，然后从远程库克隆。

首先，登录GitHub，创建一个新的仓库，名字自己取（这里取为gitskills），勾选Initialize this repository with a README，这样GitHub会自动为我们创建一个README.md文件。创建完毕后，可以看到README.md文件：

![](/克隆远程库1.png)

![](/克隆远程库2.png)

接下来用命令git clone克隆一个本地库：

**git clone  https://github.com/liutongxue123/gitskills.git**

或者

**git clone git@github.com:liutongxue123/gitskills.git**

注：Git支持多种协议，默认的git://使用ssh，但也可以使用https等其他协议。
使用https除了速度慢以外，还有个最大的麻烦是每次推送都必须输入口令，但是在某些只开放http端口的公司内部就无法使用ssh协议而只能用https。

![](/克隆远程库3.png)

如果有多个人协作开发，那么每个人各自从远程克隆一份就可以了。

**要克隆一个仓库，首先必须知道仓库的地址，然后使用git clone命令克隆。
Git支持多种协议，包括https，但通过ssh支持的原生git协议速度最快。**

### 6、分支管理

#### （1）创建与合并分支

​	每次提交，Git都把它们串成一条时间线，这条时间线就是一个分支。截止到目前，只有一条时间线，在Git里，这个分支叫主分支，即 master分支。HEAD严格来说不是指向提交，而是指向master，master才是指向提交的，所以，HEAD指向的就是当前分支。

​	一开始的时候，master分支是一条线，Git用master指向最新的提交，再用HEAD指向master，就能确定当前分支，以及当前分支的提交点：

![](/分支管理1.jpg)

每次提交，master分支都会向前移动一步，这样，随着你不断提交，master分支的线也越来越长。

当我们创建新的分支，例如dev时，Git新建了一个指针叫dev，指向master相同的提交，再把HEAD指向dev，就表示当前分支在dev上：

![](/分支管理2.jpg)

从现在开始，对工作区的修改和提交就是针对dev分支了，比如新提交一次后，dev指针往前移动一步，而master指针不变：

![](/分支管理3.jpg)

假如我们在dev上的工作完成了，就可以把dev合并到master上。Git怎么合并呢？最简单的方法，就是直接把master指向dev的当前提交，就完成了合并：

![](/分支管理4.jpg)

所以Git合并分支也很快！就改改指针，工作区内容也不变！
合并完分支后，甚至可以删除dev分支。删除dev分支就是把dev指针给删掉，删掉后，我们就剩下了一条master分支：

![](/分支管理5.jpg)

**下面是实际命令：**

**a.创建dev分支，然后切换到dev分支，最后再返回master分支：**

**\$ git checkout -b dev**
Switched to a new branch 'dev'

git checkout命令加上-b参数表示创建并切换，相当于以下两条命令：

**\$ git branch dev**

**\$ git checkout dev**

Switched to branch 'dev'

然后，用git branch命令查看当前分支：
**$ git branch**

*dev
master

git branch命令会列出所有分支，当前分支前面会标一个*号。

然后，我们就可以在dev分支上正常提交，当dev分支的工作完成后，我们就可以切换回master分支：

**$ git checkout master**

切换回master分支之后，在dev分支做出的更改等将不会存在。

![](/分支管理6.jpg)

**b.把dev分支的工作成果合并到master分支上：**

**$ git merge dev**

其执行结果为：

Updating d17efd8..fec145a
Fast-forward
readme.txt | 1 +
1 file changed, 1 insertion(+)

git merge命令用于合并指定分支到当前分支。

注意到上面的Fast-forward信息，Git告诉我们，这次合并是“快进模式”，也就是直接把master指向dev的当前提交，所以合并速度非常快。

合并完成后，就可以放心地删除dev分支了：

**$ git branch -d dev**

#### （2）解决冲突

创建新的feature1分支，当master分支和feature1分支各自都分别有新的提交。如下：

![](/分支管理7.jpg)

这种情况下，Git无法执行“快速合并”，只能试图把各自的修改合并起来，但这种合并就可能会有冲突。

git status也可以告诉我们冲突的文件。

必须手动解决冲突后再提交。修改后如图：

![](/分支管理8.jpg)

用带参数的git log也可以看到分支的合并情况：

**$ git log --graph --pretty=oneline --abbrev-commit**

#### （3）分支管理策略

通常，合并分支时，如果可能，Git会用“Fast forward”模式，但这种模式下，删除分支后，会丢掉分支信息，如下图：

![](/分支管理4.jpg)

如果要强制禁用“Fast forward”模式，就需要Git在merge时生成一个新的commit，这样，从分支历史上就可以看出分支信息。

![](/分支管理9.jpg)

接下来是--no-ff方式的merge的具体实现流程：

**a.首先创建并切换到新的dev分支：**

**git checkout -b dev**

**b.修改一个文件，并提交一个新的commit，如修改readme.txt文件：**

**git add readme.txt**
​**git commit -m "add merge"**

[dev 6224937] add merge
1 file changed, 1 insertion(+)

**c.现在切换回master：**

**git checkout master**

Switched to branch 'master'

**d.现在合并dev分支**

**git merge --no-ff -m "merge with no-ff" dev**

Merge made by the 'recursive' strategy.
readme.txt | 1 +
1 file changed, 1 insertion(+)

注意--no-ff参数，表示禁用“Fast forward”，

因为本次合并要创建一个新的commit，所以加上-m参数，把commit描述写进去。

**e.合并完成后，可以用git log查看分支历史：**

**git log --graph --pretty=oneline --abbrev-commit**

*7825a50 merge with no-ff
|\
| * 6224937 add merge
|/

*59bc1cb conflict fixed
...

#### （4）分支策略

master分支应该是非常稳定的，也就是仅用来发布新版本，平时不能在上面干活。

干活都在dev分支上，也就是说，dev分支是不稳定的，到某个时候，比如1.0版本发布时，再把dev分支合并到master上，在master分支发布1.0版本。

在团队合作中，每个人在自己的分支上干活，然后不时往dev分支合并，当有最终的版本确定时，将dev分支合并到master分支上，如下图所示：

![](/分支管理10.jpg)



#### （5）Bug分支

修复bug时，我们会通过创建新的bug分支进行修复，然后合并，最后删除；

当手头工作没有完成时，先把工作现场**git stash**一下，然后去修复bug，修复后，再**git stash pop**，回到工作现场。

当在工程中出现bug时，可以立即新建一个分支来进行修复，修复完成后，再合并到原来的分支，并删除用来修复bug的分支。

当原有工作进行到一半，临时需要去解决bug时，可以先将当前工作现成存储起来，等修复完bug后，再恢复现场继续进行原来的工作。

下面是具体教程：

**假设此时在dev分支工作**

**a.用git stash存储工作现场**

**git stash**

Saved working directory and index state WIP on dev: 6224937 add
merge
HEAD is now at 6224937 add merge

此时，用git status查看状态，就是干净的（除非有没有被Git管理的文件），因此可以放心创建bug修复分支。

**b.先确定要在哪个分支上修复bug分支，转到对应分支上，然后创建临时分支**

假如需要修复的bug在master分支

**git checkout master**

Switched to branch 'master'
Your branch is ahead of 'origin/master' by 6 commits.

**git checkout -b issue-101**

Switched to a new branch 'issue-101'

如上实现，在dev分支上创建issue-101分支，然后转到该分支下。

**c.修复bug并提交**

首先先对bug进行修改

**git add readme.txt**

**git commit -m "fix bug 101"**

**d.修复完成后，切换到bug所在原分支，并完成合并，最后删除issue-101分支**

**git checkout master**

**git merge --no-ff -m "merged bug fix 101" issue-101**

**git branch -d issue-101**

**e.返回dev分支，继续原来的工作**

**git checkout dev**

**git status**

**f.查看原来保存起来的工作现场并恢复**

**git stash list**

stash@{0}: WIP on dev: 6224937 add merge

工作现场还在，Git把stash内容存在某个地方了，但是需要恢复一下，有两个办法：

一是用**git stash apply**恢复，但是恢复后，stash内容并不删除，你需要用**git stash drop**来删除；

另一种方式是用**git stash pop**，恢复的同时把stash内容也删了。

**g.可以多次保存工作现场，然后恢复指定的stash**

**git stash list**

**git stash apply stash@{0}**

#### （6）Feature分支

​	软件开发中，总有无穷无尽的新的功能要不断添加进来。添加一个新功能时，你肯定不希望因为一些实验性质的代码，把主分支搞乱了，所以，每添加一个新功能，最好新建一个feature分支，在上面开发，完成后，合并，最后，删除该feature分支。

​	现在，你终于接到了一个新任务：开发代号为Vulcan的新功能，该功能计划用于下一代星际飞船。
于是准备开发：

**git checkout -b feature-vulcan**

5分钟后，开发完毕：

**git status**

**git add vulcan.py**

**git commit -m "add feature vulcan"**

切回dev，准备合并：

**git checkout dev**

一切顺利的话，feature分支和bug分支是类似的，合并，然后删除。

但是，就在此时，接到上级命令，因经费不足，新功能必须取消！虽然白干了，但是这个分支还是必须就地销毁：

**git branch -d feature-vulcan**

error: The branch 'feature-vulcan' is not fully merged.
If you are sure you want to delete it, run 'git branch -D featurevulcan'.

销毁失败。Git友情提醒，feature-vulcan分支还没有被合并，如果删除，将丢失掉修改，如果要强行删除，需要使用命令git branch -D feature-vulcan。

现在我们强行删除：

**git branch -D feature-vulcan**
终于删除成功！

#### （7）多人协作

当从远程仓库克隆时，实际上Git自动把本地的master分支和远程的master分支对应起来了，并且，远程仓库的默认名称是origin。

1、查看远程库的信息：

**git remote**

显示远程库更详细的信息：

**git remote -v**

2、推送分支

推送分支，就是把该分支上的所有本地提交推送到远程库。推送时，要指定本地分支，这样，Git就会把该分支推送到远程库对应的远程分支上：

**git push origin master**

如果要推送其他分支，比如dev，就改成：

**git push origin dev**

但是，并不是一定要把本地分支往远程推送，那么，哪些分支需要推送，哪些不需要呢？

- master分支是主分支，因此要时刻与远程同步；

- dev分支是开发分支，团队所有成员都需要在上面工作，所以也需要与远程同步；

- bug分支只用于在本地修复bug，就没必要推到远程了，除非老板要看看你每周到底修复了几个bug；

- feature分支是否推到远程，取决于你是否和你的小伙伴合作在上面开发。

3、抓取分支

多人协作时，大家都会往master和dev分支上推送各自的修改。

现在，模拟一个你的小伙伴，可以在另一台电脑（注意要把SSH Key添加到GitHub）或者同一台电脑的另一个目录下克隆：

**git clone git@github.com:michaelliao/learngit.git**

当你的小伙伴从远程库clone时，默认情况下，你的小伙伴只能看到本地的master分支。不信可以用git branch命令看看：

**git branch**

现在，你的小伙伴要在dev分支上开发，就必须创建远程origin的dev分支到本地，于是他用这个命令创建本地dev分支：

**git checkout -b dev origin/dev**

现在，他就可以在dev上继续修改，然后，时不时地把dev分支push到远程：

**git commit -m "add /usr/bin/env"**

**git push origin dev**

你的小伙伴已经向origin/dev分支推送了他的提交，而碰巧你也对同样的文件作了修改，并试图推送：

 **git add hello.py**
​ **git commit -m "add coding: utf-8"**
[dev bd6ae48] add coding: utf-8
1 file changed, 1 insertion(+)
 **git push origin dev**
To git@github.com:michaelliao/learngit.git
! [rejected] dev -> dev (non-fast-forward)
error: failed to push some refs to 'git@github.com:michaelliao/
learngit.git'
hint: Updates were rejected because the tip of your current branch
is behind
hint: its remote counterpart. Merge the remote changes (e.g. 'git
pull')
hint: before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for
details.

推送失败，因为你的小伙伴的最新提交和你试图推送的提交有冲突，解决办法也很简单，Git已经提示我们，先用git pull把最新的提交从origin/dev抓下来，然后，在本地合并，解决冲突，再推送：

**git pull**

remote: Counting objects: 5, done.
remote: Compressing objects: 100% (2/2), done.
remote: Total 3 (delta 0), reused 3 (delta 0)
Unpacking objects: 100% (3/3), done.
From github.com:michaelliao/learngit
fc38031..291bea8 dev -> origin/dev
There is no tracking information for the current branch.
Please specify which branch you want to merge with.
See git-pull(1) for details
git pull <remote> <branch>
If you wish to set tracking information for this branch you can do
so with:
git branch --set-upstream dev origin/<branch>

git pull也失败了，原因是没有指定本地dev分支与远程origin/dev分支的链接，根据提示，设置dev和origin/dev的链接：

**git branch --set-upstream dev origin/dev**

再pull：

**git pull**

这回git pull成功，但是合并有冲突，需要手动解决，解决的方法和分支管理中的解决冲突完全一样。解决后，提交，再push：

因此，多人协作的工作模式通常是这样：
1. 首先，可以试图用git push origin branch-name推送自己的修改；
2. 如果推送失败，则因为远程分支比你的本地更新，需要先用git pull试图合并；
3. 如果合并有冲突，则解决冲突，并在本地提交；
4. 没有冲突或者解决掉冲突后，再⽤用git push origin branch-name推送就能成功！

如果git pull提示“no tracking information”，则说明本地分支和远程分支的链接关系没有创建，用命令git branch --set-upstream branch-name origin/branch-name。

### 7、标签管理

​	发布一个版本时，我们通常先在版本库中打一个标签，这样，就唯一确定了打标签时刻的版本。将来无论什么时候，取某个标签的版本，就是把那个打标签的时刻的历史版本取出来。所以，标签也是版本库的一个快照。
​	Git的标签虽然是版本库的快照，但其实它就是指向某个commit的指针（跟分支很像，但是分支可以移动，标签不能移动），所以，创建和删除标签都是瞬间完成的。

#### （1）创建标签

1、在Git中打标签非常简单，首先，切换到需要打标签的分支上：
**git branch**

*dev
  master

**git checkout master**
Switched to branch 'master'

然后，敲命令git tag name就可以打一个新标签：

**git tag v1.0**

可以用命令git tag查看所有标签：

**git tag**

v1.0

2、默认标签是打在最新提交的commit上的，为原来的commit打标签。方法是找到历史提交的commit id，然后打上就可以了：

**git log --pretty=oneline --abbrev-commit**

6a5819e merged bug fix 101
cc17032 fix bug 101
7825a50 merge with no-ff
6224937 add merge
59bc1cb conflict fixed
400b400 & simple
75a857c AND simple
fec145a branch test
d17efd8 remove test.txt
...

比如说要对“add merge”这次提交打标签，它对应的commit id是“6224937”，敲入命令：

**git tag v0.9 6224937**

再用命令git tag查看标签：

**git tag**

v0.9
v1.0

注意，标签不是按时间顺序列出，而是按字母排序的。可以用git show tagname查看标签信息：

**git show v0.9**

commit 622493706ab447b6bb37e4e2a2f276a20fed2ab4
Author: Michael Liao <askxuefeng@gmail.com>

Date: Thu Aug 22 11:22:08 2013 +0800
add merge
...

可以看到，“v0.9”确实打在“add merge”这次提交上。

3、还可以创建带有说明的标签，用-a指定标签名，-m指定说明文字：

**git tag -a v0.1 -m "version 0.1 released" 3628164**

用命令git show tagname可以看到说明文字：

**git show v0.1**

还可以通过-s用私钥签名一个标签：

**git tag -s v0.2 -m "signed version 0.2 released" fec145a**

签名采用PGP签名，因此，必须首先安装gpg（GnuPG），如果没有找到gpg，或者没有gpg密钥对，就会报错：
gpg: signing failed: secret key not available
error: gpg failed to sign the data
error: unable to sign the tag

如果报错，请参考GnuPG帮助文档配置Key。
用命令git show tagname可以看到PGP签名信息：

**git show v0.2**

用PGP签名的标签是不可伪造的，因为可以验证PGP签名。验证签名的方法比较复杂，这里就不介绍了。

#### （2）操作标签

a. 如果标签打错，可以删除：

**git tag -d v0.1**

因为创建的标签都只存储在本地，不会自动推送到远程。所以，打错的标签可以在本地安全删除。

b.推送某个标签到远程:

**git push origin tagname**

或者，一次性推送全部尚未推送到远程的本地标签：

**git push origin --tags**

c.删除远程标签：

先从本地删除

**git tag -d v0.9**

再从远程删除

git push origin :refs/tags/v0.9

要看看是否真的从远程库删除了标签，可以登陆GitHub查看。

### 8、GitHub的使用

以下讲述如何在GitHub上参与一个开源项目。

比如人气极高的bootstrap项目，这是一个非常强大的CSS框架，其地址为：https://github.com/twbs/bootstrap

点“Fork”就在自己的账号下克隆了一个bootstrap仓库，然后，从自己的账号下clone：

**git clone git@github.com:liutongxue123/bootstrap.git**

一定要从自己的账号下clone仓库，这样你才能推送修改。

Bootstrap的官方仓库twbs/bootstrap、本人在GitHub上克隆的仓库my/bootstrap，以及自己克隆到本地电脑的

仓库，他们的关系就像下图显示的那样：

![](/GitHub使用.jpg)

如果想修复bootstrap的一个bug，或者新增一个功能，立刻就可以开始干活，干完后，往自己的仓库推送。

如果希望bootstrap的官方库能接受你的修改，你就可以在GitHub上发起一个pull request。当然，对方是否接受pull request就不一定了。

### 9、自定义Git

#### （1）配置姓名和邮箱地址

**git config --global user.name "Your Name"**

 **git config --global user.email "email@example.com"**

#### （2）让Git显示颜色

**git config --global color.ui true**

#### （3）忽略特殊文件

在Git工作区的根目录下创建一个特殊的.gitignore文件，然后把要忽略的文件名填进去，Git就会自动忽略这些文件。不需要从头写.gitignore文件，GitHub已经为我们准备了各种配置文件，只需要组合一下就可以使用了。

配置文件地址：https://github.com/github/gitignore

忽略文件编写原则：

1、忽略操作系统自动生成的文件，比如缩略图等；

2、忽略编译生成的中间文件、可执行文件等，也就是如果一个文件是通过另一个文件自动生成的，那自动生成的文件就没必要放进版本库，比如Java编译产生的.class文件；

3、忽略你自己的带有敏感信息的配置文件，比如存放口令的配置文件。

举例：

假设在Windows下进行Python开发，Windows会自动在有图片的目录下生成隐藏的缩略图文件，如果有自定义目录，目录下就会有Desktop.ini文件，因此你需要忽略Windows自动生成的垃圾文件：

\#Windows:

Thumbs.db
ehthumbs.db
Desktop.ini
然后，继续忽略Python编译产生的.pyc、.pyo、dist等文件或目录：

\#Python:

*.py[cod]
*.so
*.egg
*.egg-info
dist
build

加上你自己定义的文件，最终得到一个完整的.gitignore文件，内容如下：

\#My configurations:

db.ini
deploy_key_rsa

检验.gitignore的标准是git status命令是不是说“working directory clean”。

![](/git忽略文件.png)

#### （4）配置别名

给命令配置简单的别名，举例如下：

**git config --global alias.st status**

**git config --global alias.co checkout**

**git config --global alias.ci commit**

**git config --global alias.br branch**

**git config --global alias.unstage 'reset HEAD'**

### 10、搭建Git服务器

​	GitHub就是一个免费托管开源代码的远程仓库。但是对于某些视源代码如生命的商业公司来说，既不想公开源代码，又舍不得给GitHub交保护费，那就只能自己搭建一台Git服务器作为私有仓库使用。

​	搭建Git服务器需要准备一台运行Linux的机器，强烈推荐用Ubuntu或Debian，这样，通过几条简单的apt命令就可以完成安装。

首先取得用户权限：

a.安装git

**sudo apt-get install git**

b.创建一个git用户，用来运行git服务：

**sudo adduser git**

c.创建证书登录：

收集所有需要登录的用户的公钥，就是他们自己的id_rsa.pub文件，把所有公钥导入到
/home/git/.ssh/authorized_keys文件里，一行一个。

d.初始化Git仓库：

先选定一个目录作为Git仓库，假定是/srv/sample.git，在/srv目录下输入命令：

**sudo git init --bare sample.git**

Git就会创建一个裸仓库，裸仓库没有工作区，因为服务器上的Git仓库纯粹是为了共享，所以不让用户直接登录到服务器上去改工作区，并且服务器上的Git仓库通常都以.git结尾。然后，把owner改为git：

**sudo chown -R git:git sample.git**

e.禁用shell登录：

出于安全考虑，第二步创建的git用户不允许登录shell，这可以通过编辑/etc/passwd文件。找到类似下面的一行：
git​:x :​1001:1001:,,,:/home/git:/bin/bash
改为：
git:x :1001:1001:,,,:/home/git:/usr/bin/git-shell

这样，git用户可以正常通过ssh使用git，但无法登录shell，因为我们为git用户指定的gitshell每次一登录就自动退出。

f.克隆远程仓库：

**git clone git@server:/srv/sample.git**

要方便管理公钥，用Gitosis工具；
要像SVN那样变态地控制权限，用Gitolite工具。