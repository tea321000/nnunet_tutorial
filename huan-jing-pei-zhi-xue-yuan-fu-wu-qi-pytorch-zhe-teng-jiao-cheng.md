# 环境配置——pytorch折腾教程

经过半个月的折腾，我大致总结出两个办法：1、**使用学院服务器集群`module`管理器的`CUDA9.0`对pytorch源码进行编译**（缺点：最高只能到pytorch1.5，pytorch1.6需要`CUDA9.2`以上版本的编译，而且pytorch1.6原生支持nnunet的混合精度训练，不然必须额外下载安装令人头疼的`apex`）；2、学院服务器`nvidia-smi`显示的显卡驱动版本`390.46`经[查询](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)是不支持`CUDA10`的，因此`module`管理器里的`cuda10.0`也是用不了的（服务器里有什么module可以在`/cm/shared/app`路径下查看），**但我们可以自行安装**[**CUDA9.2**](https://developer.nvidia.com/cuda-92-download-archive)**和**[**CUDNN7.6.5**](https://developer.nvidia.com/rdp/cudnn-archive)（CUDNN跟CUDA版本需要对应）**对pytorch最新版本（支持CUDA9.2）进行编译**，但由于没有`root`权限，因此必须下载`run`版本和`tgz`版本文件，解压cuda和cudnn后手动添加环境变量到`PATH`和`LD_LIBRARY_PATH`这两个环境变量中而无法直接进行安装。

附学院服务器module管理器的常用命令：

```bash
#列出服务器中已经安装的module
module avail
#列出目前使用的module
module list
#添加module(服务器里有什么module可以在/cm/shared/apps路径下查看)
module add xxx
#删除module
module remove xxx
```

{% hint style="info" %}
目前为止pytorch只支持python3.6及3.7版本进行编译，因此Ubuntu20.04系统可能需要卸载系统自带的3.8安装3.7（尽量不要使用`update-alternatives`等切换Python版本，可能会出现找不到包等情况），而且在安装3.7时最好切换到`root`而不直接使用`sudo`进行安装：
{% endhint %}

```bash
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo su
apt install python3.7
```

## 方法一：使用学院的CUDA9.0进行编译

### **对虚拟环境的创建**

虚拟环境不建议使用anaconda冗余的虚拟环境（nnunet强烈不建议 因为会导致问题的产生）而建议使用`virtualenv`，anaconda具有`conda`和`pip`两个安装源，两个安装包管理之间无法相互管理另一个源安装的包，安装的package版本号也往往不同，因此常常会出现冲突等问题。但学院服务器没有`root`权限，限制只能通过

```bash
conda env create -n env_name python=3.7 
```

安装Python3.7，而本地的是Python3.6，而

```bash
python3 -m venv env_name
```

无法自定义python的版本，因此需要创建Python3.7时可以先通过`conda env create` 创建带有python3.7的conda环境`conda_env_name`，然后再在进入conda虚拟环境的情况下运行上面的指令安装python3.7的虚拟环境。

或者也可以在进入conda虚拟环境的情况下通过`conda env list`查看这个环境中python3.7的位置并使用`pip install virtualenv`安装`virtualenv`，接着

```bash
virtualenv --python=/home/user0xx/.conda/envs/conda_env_name/bin/python3.7 new_env_name
```

来创建`virtualenv`环境`new_env_name`。

{% hint style="warning" %}
通过`virtualenv`方式安装的虚拟环境在编译某些特定`package`时可能会存在问题。假如存在问题 还是建议使用`python3 -m venv`的方式创建虚拟环境
{% endhint %}



由于本人需要使用高版本的pytorch\(&gt;=1.4\)，而学院服务器是老旧的CUDA9.0，因此并没有&gt;=1.4版本的pre-build二进制pip package可以下载，因此需要from source编译pytorch。

又由于学院服务器只有GCC7（CUDA9.0只能用GCC6以下进行编译）和GCC5.5（此版本编译pytorch会有bug），因此我们需要自行编译一个GCC5.4用于源码编译。

### 编译GCC5.4

首先在[GNU release](https://gcc.gnu.org/releases.html) 页面将GCC5.4下载下来，解压后以文本打开`gcc-5.4.0/contrib/download_prerequisites`，可以发现GCC依赖于`gmp mpc mpfr`这三个package，而且该文件里会标明使用的版本号。虽然也可以直接运行

```bash
./contrib/download_prerequisites
```

但往往由于网络原因或者服务器更换等原因不能成功，建议还是在[GNU](https://ftp.gnu.org/gnu/mpfr/)上将对应版本号的package下下来解压然后放到gcc-5.4.0文件夹中，并仿照`download_prerequisites`使用

```bash
ln -sf mpfr-2.4.2 mpfr
```

对三个package都进行链接。接着因为不能直接在源码中编译创建`objdir`作为编译路径，然后进行编译（gcc全称为gnu compiler collection，可以编译C JAVA等语言，可以写all，但会耗时，更多内容参考官方文档）：

```bash
cd ..
mkdir objdir
cd objdir
../gcc-5.4.0/configure --prefix=$HOME/gcc-5.4.0 --disable-checking --enable-languages=c,c++ --disable-multilib --enable-threads=posix
make
make install
```

等待GCC漫长的编译过程。最后删除objdir即可：

```bash
rm -rf ~/objdir
```

虽然至此我们已经no root完成了对GCC的编译，但也正是因为是no root，因此我们还要手动将我们编译的GCC放到环境变量`PATH`和`LD_LIBRARY_PATH` 中，并置于系统原先的GCC路径之前，这样在寻找GCC时会首先使用我们刚刚编译的GCC而不是系统原来的GCC：

```bash
export PATH=$HOME/gcc-5.4.0/bin:PATH
export LD_LIBRARY_PATH=$HOME/gcc-5.4.0/lib:$HOME/gcc-5.4.0/lib64:$LD_LIBRARY_PATH
```

此时仅在当前的terminal生效，一劳永逸要将以上两行放到`~/.bashrc`的最后，然后`source ~/.bashrc`

### 编译pytorch1.5\(CUDA9.0支持的最高版本\)

在编译完成后，需要再从源码编译[pytorch](https://github.com/pytorch/pytorch/tree/v1.5.0)**。**由于使用的是cuda9.0，因此不支持pytorch1.6以上，只能选择pytorch1.5版本。clone时需要clone `v1.5`的tag而不是最新的`master` branch：

```bash
cd ~
#据说1.5版本就是不能用C++版本的apex会报runtime error 需要使用C++版本的apex可以编译下载1.4版本的
git clone --depth 1 --branch v1.5.0 https://github.com/pytorch/pytorch/
```

大部分都是按照github的教程来走，只是将conda install换成pip install：

```bash
pip install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi
cd pytorch
git submodule sync
git submodule update --init --recursive
#设置cmake目录
export CMAKE_PREFIX_PATH="$HOME/env_name/bin/"
#设置cudnn目录
export CUDNN_LIB_DIR="/cm/shared/apps/cudnn/7.0/lib64/"
export CUDNN_INCLUDE_DIR="/cm/shared/apps/cudnn/7.0/include/"
python setup.py install
```

尤其要注意倒数第二行不能直接复制官方github里的命令（因为是面向conda的）而要将`CMAKE_PREFIX_PATH`设置为你virtualenv的`bin`文件夹，即可以通过你自己的virtualenv找到`cmake`。或者也可以通过在[cmake官网](https://cmake.org/download/)上下载cmake的二进制预编译.sh文件，将其解压到home目录下面以后`export CMAKE_PREFIX_PATH=$HOME/cmake:$PATH` 使用下载的cmake来进行编译。

install后等待漫长的编译，pytorch也就编译成功啦。

\`\`

### 安装nnUNet

最后则是对[nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/v1.5.1)的安装**（推荐从源码安装，方便魔改代码，对nnUNet文件夹的py文件进行更新后可以实时反应到环境中 不需要再次`pip uninstall pip install`）**。PyTorch1.6以后自带混合精度训练，不需要再进行[`apex`](https://github.com/NVIDIA/apex)的安装，但由于我们这个是1.5的版本，还是需要进行安装：

```bash
cd env_name
git clone --depth 1 --branch v1.5.1 https://github.com/MIC-DKFZ/nnUNet/
git clone https://github.com/NVIDIA/apex
#先安装apex的依赖
git clone https://github.com/NVIDIA/PyProf
cd PyProf
pip install .
cd ../apex
#下面这条是C++版本的安装 我测试了没有成功 不成功可以安装Python版本 虽然效率可能只有百分之90但也足够了
#据说1.5版本就是不能用C++版本会报runtime error 需要使用可以编译下载1.4版本的
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
#Python版本
pip install -v --no-cache-dir ./
cd ../nnUNet
pip install -e .
#hiddenlayer 可选 用来显示网路拓扑图
pip install --upgrade git+https://github.com/nanohanno/hiddenlayer.git@bugfix/get_trace_graph#egg=hiddenlayer

```

{% hint style="info" %}
可以尝试拉一个旧的apex分支，可能可以安装C++版本的apex，本人未亲自测试
{% endhint %}

### 配置nnunet

安装好nnunet后还要对其进行一些文件夹路径的配置：

```bash
cd ..
#创建nnUNet数据集文件夹
mkdir dataset && cd dataset
#创建预训练 原始 训练模型三个文件夹
mkdir preprocessed raw trained_models
#在原始文件夹中创建原始数据 裁剪后数据两个文件夹
cd raw && mkdir raw_data cropped_data
```

接着修改`.bashrc`文件，在最后加上（视自己的具体目录）：

```bash
export nnUNet_raw_data_base="/home/user026/zzq/nnunet/dataset/raw"
export nnUNet_preprocessed="/home/user026/zzq/nnunet/dataset/preprocessed"
export RESULTS_FOLDER="/home/user026/zzq/nnunet/dataset/trained_models"
```

终于我们的环境都配置好了= =

## ~~方法二：更改CUDA版本~~

非常不幸由于学院服务器的显卡驱动版本太低，最高只能支持`CUDA9.1`，因此即使用no root方法安装了`CUDA9.2`，`pip install`或者from source build的pytorch也不能正常使用CUDA。因此此方法只做存档以备参考。

~~\`\`~~[~~`CUDA.run`~~](https://developer.nvidia.com/cuda-92-download-archive)~~文件直接使用`sh CUDA.run`命令安装即可，可以不使用sudo，在安装时不要选择安装显卡驱动只安装toolkit到home目录下的某个文件夹，而是对于~~[~~`cudnn.tgz`~~](https://developer.nvidia.com/rdp/cudnn-archive)~~则是按照官方的tutorial解压后复制相应的文件到CUDA的目录下即可。最后我们为防止原来的CUDA和cudnn对新的产生影响，将原来的`module`去掉后将解压的CUDA添加到环境变量中：~~

```bash
module rm cuda90
module rm cudnn
export PATH=$HOME/lib/cuda-9.2/bin:$PATH
export CPATH="$HOME/lib/cuda-9.2/include"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/lib/cuda-9.2/lib64
```

~~至此no root的CUDA和cudnn安装成功。其余步骤与方法一一样，通过编译安装GCC和pytorch；对于存在对应CUDA版本的pip二进制wheel，可以直接使用`pip install`的方法来安装pytorch。~~

