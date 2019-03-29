# PYNQ SD Card

The PYNQ image for supported boards are provided precompiled as downloadable SD card images, so you do not need to rerun this flow for these boards unless you want to make changes to the image flow.

This flow can also be used as a starting point to build a PYNQ image for another Zynq / Zynq Ultrascale board.

这个流程被用于为其他zynq板卡制作PYNQ启动镜像。

The image flow will create the BOOT.bin, the u-boot bootloader, the Linux Device tree blob, and the Linux kernel.

The source files for the PYNQ image flow build can be found here:

```
<PYNQ repository>/sdbuild
```

More details on configuring the root filesystem can be found in the README file in the folder above.

## Prepare the Building Environment

It is recommended to use a Ubuntu OS to build the image. If you do not have a Ubuntu OS, you may need to prepare a Ubuntu virtual machine (VM) on your host OS. 

We provide in our repository a *vagrant* file that can help you install the Ubuntu VM on your host OS.

提供了一个vagrant file，是在主机上安装ubuntu虚拟的教程

If you do not have a Ubuntu OS, and you need a Ubuntu VM, do the following:

> 1. Download the [vagrant software](https://www.vagrantup.com/) and the [Virtual Box](https://www.virtualbox.org/). Install them on your host OS.
>
> 2. In your host OS, open a terminal program. Locate your PYNQ repository, where the vagrant file is stored.
>
>    ```
>    cd <PYNQ repository>
>    ```
>
> 3. You can then prepare the VM using the following command. This step will prepare a Ubuntu VM called *pynq_vm* on your Virtual Box. The Ubuntu packages on the VM will be updated during this process; the Ubuntu desktop will also be installed so you can install Xilinx software later.
>
>    ```
>    vagrant up
>    ```
>
>    After the VM has been successfully loaded, you will see a folder */pynq* on your VM; this folder is shared with your PYNQ repository on your host OS.
>
> 4. (optionally) To restart the VM without losing the shared folder, in your terminal, run:
>
>    ```
>    vagrant reload
>    ```
>
> 5. Now you are ready to install Xilinx tools. You will need PetaLinux, Vivado, and SDx for building PYNQ image. The version of Xilinx tools for each PYNQ release is shown below:
>
>    | Release version | Xilinx Tool Version |
>    | --------------- | ------------------- |
>    | v1.4            | 2015.4              |
>    | v2.0            | 2016.1              |
>    | v2.1            | 2017.4              |
>    | v2.2            | 2017.4              |
>    | v2.3            | 2018.2              |
>    | v2.4            | 2018.3              |

已经有虚拟机的情况下，作如下操作：

If you already have a Ubuntu OS, you can do the following:

> 1. Install dependencies using the following script. This is necessary if you are not using our vagrant file to prepare the environment.
>
>    ```
>    <PYNQ repository>/sdbuild/scripts/setup_host.sh
>    ```
>
> 2. Install correct version of the Xilinx tools, including PetaLinux, Vivado, and SDx. See the above table for the correct version of each release.

## Building the Image   建立镜像

Once you have the building environment ready, you can start to build the image following the steps below:

环境准备好之后，进行如下操作：

> 1. Source the appropriate settings files from PetaLinux, Vivado, and SDx.  初始化xilinx的工具
> 2. Navigate to the following directory and run make   编译源码
>
> ```
> cd <PYNQ repository>/sdbuild/
> make
> ```

The build flow can take several hours. By default images for all of the supported boards will be built.

## Retargeting to a Different Board

**重新定位到另一个板卡**

Additional boards are supported through external（外部的）  *board repositories*.

**通过外部的 board目录，可以支持别的板卡**

 A board repository consists of a directory for each board consisting of a spec file and any other files.

**板卡目录包含每个板的目录，包括spec文件和任何其他文件。**

 The board repository is treated the same way as the `<PYNQ repository>/boards` directory.

**这些自己建立的目录，与PYNQ/board文件下官方的文件一样使用**

### Elements of the specification file

The specification file should be name `<BOARD>.spec` where BOARD is the name of the board directory. A minimal spec file contains the following information

```
ARCH_${BOARD} := arm
BSP_${BOARD} := mybsp.bsp
BITSTREAM_${BOARD} := mybitstream.bsp
```

where `${BOARD}` is also the name of the board. The ARCH should be *arm* for Zynq-7000 or *aarch64*for Zynq UltraScale+. 

If no bitstream is provided then the one included in the BSP will be used by default. 

**如果没有提供比特流，则默认使用BSP中包含的比特流。**

All paths in this file should be relative to the board directory. 

To customise(定制) the BSP a `petalinux_bsp` folder can be included in the board directory the contents of which will be added to the provided BSP before the project is created.

**要定制BSP，可以在board目录中添加`petalinux_bsp`文件夹，其内容将在创建项目之前添加到提供的BSP中。**

 See the ZCU104 for an example of this in action.    以ZCU104为例

This is designed to allow for additional drivers, kernel or boot-file patches and device tree configuration that are helpful to support elements of PYNQ to be added to a pre-existing BSP.

**这种设计有助于支持将PYNQ元素添加到预先存在的BSP的其他驱动程序，内核或引导文件修补程序和设备树配置。**

If a suitable PetaLinux BSP is unavailable for the board then this can be left blank; in this case, an HDF file needs to be provided in the board directory. The *system.hdf* file should be placed in the `petalinux_bsp/hardware_project` folder and a new generic BSP will be created as part of the build flow.

**如果PetaLinux BSP不适用于板卡，则可以将其空置。  在这种情况下，需要在board目录中提供HDF文件。即将system.hdf文件放到路径petalinux_bsp/hardware_project下，然后将创建一个新的通用BSP作为构建流程的一部分。**

### Board-specific packages



A `packages` directory can be included in board directory with the same layout as the `<PYNQ repository>/sdbuild/packages` directory. 

**一个`packages`目录可以包含在board目录中，其布局与`<PYNQ repository> / sdbuild / packages`目录相同。**

Each subdirectory is a package that can optionally be installed as part of image creation.

 See `<PYNQ repository>/sdbuild/packages/README.md` for a description of the format of a PYNQ sdbuild package.

每个子目录都是一个package，可以选择作为映像创建的一部分安装。

可以看 <PYNQ repository> /sdbuild/packages/README.md 

To add a package to the image you must also define a `STAGE4_PACKAGE_${BOARD}` variable in your spec file. 

**要向image添加package，还必须在spec文件中定义“STAGE4_PACKAGE _ $ {BOARD}”变量。**

These can either packages in the standard sdbuild library or ones contained within the board package.

**这些可以在标准sdbuild库中打包，也可以在board包中包含。**

 It is often useful to add the `pynq` package to this list which will ensure that a customised PYNQ installation is included in your final image.

**将`pynq`包添加到此列表中通常很有用，这将确保最终映像中包含自定义PYNQ安装。**

### Using the PYNQ package

The `pynq` package will treat your board directory the same as any of the officially supported boards.

`pynq`软件包会将您的board目录视为与任何官方支持的电路板相同。

 This means, in particular, that:

> 1. A `notebooks` folder, if it exists, will be copied into the `jupyter_notebooks` folder in the image. Notebooks here will overwrite any of the default ones.
>
> **一个notebooks`文件夹（如果存在）将被复制到图像中的`jupyter_notebooks`文件夹中。**这里的notebooks将会覆盖默认值。
>
> 1. Any directory containing a bitstream will be treated as an overlay and copied into the overlays folder of the PYNQ installation. Any notebooks will likewise by installed in an overlay-specific subdirectory.
>
> 1任何包含比特流的目录都将被视为overlay并复制到PYNQ安装的覆盖文件夹中。任何notebooks同样都安装在特定于覆盖的子目录中。

## Building from a board repository

To build from a third-party board repository pass the `${BOARDDIR}` variable to the sdbuild makefile.

```
cd <PYNQ repository>/sdbuild/
make BOARDDIR=${BOARD_REPO}
```

The board repo should be provided as an absolute path. 

The `${BOARDDIR}` variable can be combined with the `${BOARD}` variable if the repository contains multiple boards and only a subset should be built.

如果存储库包含多个板并且只应构建一个子集，则`$ {BOARDDIR}`变量可以与`$ {BOARD}`变量组合。

[Next ](https://pynq.readthedocs.io/en/latest/pynq_package.html)[ Previous](https://pynq.readthedocs.io/en/latest/overlay_design_methodology/overlay_tutorial.html)