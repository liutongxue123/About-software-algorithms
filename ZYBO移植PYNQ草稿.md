# ZYBO移植PYNQ草稿

在zynq上跑Linux，需要以下文件：

BOOT image (BOOT.bin)

kernel image (uImage)

devicetree blob (devicetree.dtb)

rootfs

1、准备环境

设置bash环境来使用xilinx SDK tools，

```
source /opt/Xilinx/SDK/2016.3/settings64.sh
source /opt/Xilinx/Vivado/2016.3/settings64.sh    #这两条相当于初始化SDK和vivado

#对于2017.1版本以前的SDK，执行以下命令
export CROSS_COMPILE=arm-xilinx-linux-gnueabi-
#对于2017.1版本以前的SDK，执行以下命令
export CROSS_COMPILE=arm-linux-gnueabihf-gcc-

```

不需要自己为SD卡分区，烧录PYNQ的镜像到SD卡时，会自动分区完毕

完成的分区为：

Partition-1:

- Type: fat32
- Free Space Preceding: 4MB (IMPORTANT!!!)
- Size: 52MB
- Label: BOOT (Optional)

Partition-2:

- Type: ext4
- Size: whatever is left
- Label: rootfs (Optional)

注意事项：

- 先解除挂载UNMOUNTING，然后再移除SD卡

2、制作BOOT.bin文件











