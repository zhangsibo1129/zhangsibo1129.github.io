---
title: ONNX Runtime 项目编译
author: zhang
date: 2023-06-08 15:00:00 +0800
categories: [Blogging, ONNX Runtime]
tags: [onnxruntime]
---

## 基础环境搭建

### 1. 准备服务器

在华为云 ECS 服务申请如下配置

- 实例类型：AI加速型ai1s
- 规格名称：ai1s.large.4
- 操作系统：Ubuntu 18.04

### 2. 安装驱动

卸载旧驱动，服务器自带驱动较老，需更新

```bash
cd /usr/local/Ascend/opp/aicpu/script
./uninstall.sh

cd /usr/local/Ascend/ascend-toolkit/latest/toolkit/script
./uninstall.sh

cd /usr/local/Ascend/driver/script
./uninstall.sh
```

安装最新驱动，在官网下载最新 NPU 驱动，型号为 [Atlas 300I 推理卡（型号：3010）](https://www.hiascend.com/zh/hardware/firmware-drivers/community?product=2&model=3&cann=6.3.RC2.alpha002&driver=1.0.18.alpha) 

```bash
# 添加执行权限
chmod u+x A300-3010-npu-driver_6.0.0_linux-x86_64.run

# 安装驱动
./A300-3010-npu-driver_6.0.0_linux-x86_64.run --full

# 显示 NPU 加速卡信息
npu-smi info

# 若能正常显示两块 Ascend 310，则安装成功
```

### 3. 安装开发套件

安装 Anaconda 

```bash
# 下载安装包
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh

# 添加可执行权限
chmod u+x Anaconda3-2023.03-1-Linux-x86_64.sh

# 执行安装，安装过程中选择执行 conda init
./Anaconda3-2023.03-1-Linux-x86_64.sh

# 更新 conda
conda update -n base -c defaults conda

# 重新打开终端进入base环境，并创建虚拟环境
conda create -n onnxruntime python=3.8

# 激活环境
conda activate onnxruntime
```

安装依赖软件

```bash
# 安装依赖软件
apt-get install -y gcc g++ make cmake zlib1g zlib1g-dev openssl libsqlite3-dev libssl-dev libffi-dev unzip pciutils net-tools libblas-dev gfortran libblas3 libopenblas-dev
```

安装 CANN 开发套件，在[官网](https://www.hiascend.com/zh/software/cann/community)选择合适版本

```bash
# 下载安装包
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Florence-ASL/Florence-ASL%20V100R001C30SPC702/Ascend-cann-toolkit_6.3.RC2.alpha002_linux-x86_64.run

# 添加可执行权限
chmod u+x Ascend-cann-toolkit_6.3.RC2.alpha002_linux-x86_64.run

# 安装
./Ascend-cann-toolkit_6.3.RC2.alpha002_linux-x86_64.run  --install
```

## 项目编译

### 1. 配置编译环境

安装 cmake-3.26 或更高版本

```bash
python3 -m pip install cmake
```

升级 gcc 和 g++ 版本

```bash
# 添加 PPA 源
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update

# 安装高版本
sudo apt-get install gcc-11 # 大版本号即可
sudo apt-get install g++-11

# 删除原链接
cd /usr/bin
rm /usr/bin/gcc
rm /usr/bin/g++

# 添加软连接
ln -s gcc-11 gcc
ln -s g++-11 g++
```

安装 Python 依赖库

```bash
conda install attrs numpy decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests
```

添加 Ascend 相关环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 2. 编译构建

```bash
# 克隆项目
git clone --recursive https://github.com/Microsoft/onnxruntime.git
cd onnxruntime

# 编译
./build.sh --allow_running_as_root --config RelWithDebInfo --build_shared_lib --parallel --use_cann --build_wheel
```
 
若想在后台运行编译过程

```bash
# 后台编译
setsid ./build.sh --allow_running_as_root --config RelWithDebInfo --build_shared_lib --parallel --use_cann --build_wheel &>../build.log &

# 实时监控日志
tail -f ../build.log
```

若需要部署应用，可以将编译后的动态链接库和头文件进行安装

```bash
# 安装
make -C build/Linux/RelWithDebInfo/ install

# 卸载（不删除文件夹）
cat install_manifest.txt | sudo xargs rm
```
