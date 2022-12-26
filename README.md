# 代码说明

由于所有文件较大，而一直存在网络问题，GitHub 始终无法上传，因此将文件上传至如下链接的云盘中，带来不便还望谅解！

https://cloud.tsinghua.edu.cn/d/4d975db60e4141f4b831/

# 功能简介

本次大作业中，用 C++ 实现了 CNN 网络的训练和推理流程。具体地，实现了卷积层、全连接层及若干种类的激活层、池化层的推理和实现，并实现了 LeNet、AlexNet、RegNet18、VGG11 四种典型的网络。同时，程序中也预留了各种接口，便于添加其他种类的层和网络类型。所有代码均经过了 PyTorch 程序验证。

# 文件列表

| 文件名                    | 说明                              |
| ------------------------- | --------------------------------- |
| `build`                   | 空文件夹，用于存储 Cmake 编译文件 |
| `include`                 | 声明文件                          |
| $~~$├ `global_include.h`  | 全局引用声明                      |
| $~~$├ `tensor.h`          | 张量类声明                        |
| $~~$├ `layer.h`           | 神经网络层类声明                  |
| $~~$├ `network.h`         | 神经网络类声明                    |
| $~~$└ `tools.h`           | 功能模块声明                      |
| `src`                     | 定义文件                          |
| $~~$├ `tensor`            | 张量类定义                        |
| $~~~~~~$├ `tensor1d.cpp`  | 一维张量类定义                    |
| $~~~~~~$├ `tensor2d.cpp`  | 二维张量类定义                    |
| $~~~~~~$├ `tensor3d.cpp`  | 三维张量类定义                    |
| $~~~~~~$└ `tensor4d.cpp`  | 四维张量类定义                    |
| $~~$├ `layer`             | 神经网络层类定义                  |
| $~~~~~~$├ `conv.cpp`      | 卷积层类定义                      |
| $~~~~~~$├ `fc.cpp`        | 全连接层类定义                    |
| $~~~~~~$├ `ReLU1d.cpp`    | 一维 ReLU 激活层类定义            |
| $~~~~~~$├ `ReLU3d.cpp`    | 三维 ReLU 激活层类定义            |
| $~~~~~~$├ `sigmoid1d.cpp` | 一维 Sigmoid 激活层类定义         |
| $~~~~~~$├ `sigmoid3d.cpp` | 三维 Sigmoid 激活层类定义         |
| $~~~~~~$├ `avgpool.cpp`   | 平均池化层类定义                  |
| $~~~~~~$├ `maxpool.cpp`   | 最大池化层类定义                  |
| $~~~~~~$└ `res.cpp`       | 残差块类定义                      |
| $~~$├ `network`           | 神经网络类定义                    |
| $~~~~~~$├ `network.cpp`   | 神经网络类定义                    |
| $~~~~~~$├ `LeNet.cpp`     | LeNet 网络类定义                  |
| $~~~~~~$├ `AlexNet.cpp`   | AlexNet 网络类定义                |
| $~~~~~~$├ `ResNet18.cpp`  | ResNet18 网络类定义               |
| $~~~~~~$└ `VGG11.cpp`     | VGG11 网络类定义                  |
| $~~$└ `tools.cpp`         | 功能模块定义                      |
| `python`                  | 辅助文件及数据储存                |
| $~~$├ `data_gem.ipynb`    | 验证数据生成程序                  |
| $~~$├ `forward_val.ipynb` | 推理结果验证程序                  |
| $~~$├ `tools.py`          | 功能模块                          |
| $~~$├ `LeNet_bin_data`    | LeNet 网络验证数据                |
| $~~$├ `AlexNet_bin_data`  | AlexNet 网络验证数据              |
| $~~$├ `ResNet18_bin_data` | ResNet18 网络验证数据             |
| $~~$└ `VGG11_bin_data`    | VGG11 网络验证数据                |
| `main.cpp`                | 推理结果验证程序                  |

# 模块介绍

## 张量类

![image](https://github.com/llc0126/c-unix_hw_llc/blob/main/fig/tensor.svg)

张量虚基类模板的声明在 `include/tensor.h` 中，如：

```c++
template <class dtype>
class tensor
{
public:
    int *shape;
    dtype *data;
    ~tensor()
    {
        delete[] data;
        delete[] shape;
        return;
    };
    virtual void print() = 0;
    virtual void load(char *, char *) = 0;
};
```

张量虚基类模板拥有 2 个成员，分别为张量各维度大小数组的指针和张量数据的指针。另有 3 个方法，分别为析构、在标准显示上输出和从文件读取。

不同维度的张量类模板由张量虚基类模板继承得到，其声明在 `include/tensor.h` 中，定义分别在

- `src/tensor/tensor1d.cpp`
- `src/tensor/tensor2d.cpp`
- `src/tensor/tensor3d.cpp`
- `src/tensor/tensor4d.cpp`

中，对应一维至四维的不同张量。不同维度的张量类模板除了张量虚基类模板的成员和方法外，还重载了 `()` 运算符用于快速读写数组数据。以一维张量类模板为例，上述重载的声明为：

```c++
template <class dtype>
class tensor1d : public tensor<dtype>
{
public:
/* ...... */
    dtype &operator()(int);
    const dtype &operator()(int) const;
/* ...... */
};
```

在 `include/tensor.h` 中，还实现了在三维和一维张量间互相转换的函数：

```c++
template <class dtype>
tensor1d<dtype> *flatten(tensor3d<dtype> *input, int s0, int s1, int s2)
/* ...... */

template <class dtype>
tensor3d<dtype> *unflatten(tensor1d<dtype> *input, int s0, int s1, int s2)
/* ...... */
```

## 神经网络层类

![image](https://github.com/llc0126/c-unix_hw_llc/blob/main/fig/layer.svg)

神经网络层虚基类的声明在 `include/layer.h` 中，如：

```c++
class layer
{
public:
    virtual void clean_partial() = 0;
    virtual void update(double) = 0;
    virtual void print_param() = 0;
};
```

神经网络虚基类拥有 3 个方法，分别为清除梯度、权重更新和在标准显示上输出参数。在此基础上继承出对应三维和一维输入的两个神经网络层虚基类，同样声明在 `include/layer.h` 中，如：

```c++
class layer1d : public layer
{
public:
    virtual tensor1d<double> *forward(tensor1d<double> *) = 0;
    virtual void clean_partial() = 0;
    virtual tensor1d<double> *backward(tensor1d<double> *) = 0;
    virtual void update(double) = 0;
    virtual void print_param() = 0;
};

class layer3d : public layer
{
public:
    virtual tensor3d<double> *forward(tensor3d<double> *) = 0;
    virtual void clean_partial() = 0;
    virtual tensor3d<double> *backward(tensor3d<double> *) = 0;
    virtual void update(double) = 0;
    virtual void print_param() = 0;
};
```

与其基类相比，增加了不同维度下的前后向计算方法。

不同神经网络层类由神经网络层虚基类继承得到，其声明在 `include/layer.h` 中，定义在 `src/layer` 文件夹中。与其基类相比，不同神经网络层类按照神经网络层种类，增加了指示大小的成员、权重及梯度张量指针成员或中间数据张量指针成员，如 `res` 残差层的特殊层还包含了其内层神经网络层指针成员。同时，这些类声明和实现了对应的构造和析构方法。

## 神经网络类


![image](https://github.com/llc0126/c-unix_hw_llc/blob/main/fig/network.svg)

神经网络层虚基类的声明在 `include/network.h` 中，如：

```c++
class network
{
public:
    int num_layers3d;
    int num_layers1d;
    layer3d **layers3d;
    layer1d **layers1d;
    int flatten_size[3];
    ~network();
    tensor1d<double> *forward(tensor3d<double> *, bool = false);
    void clean_partial();
    void backward(tensor1d<double> *);
    void update(double);
    virtual void load(char *) = 0;
};
```

神经网络虚基类拥有 5 个成员，分别为三维、一维的神经网络层数和神经网络层指针指针，以及三维向一维转换的大小参数。拥有 6 个方法，其定义在 `src/network/network.cpp` 中，依次为：

- 析构
- 前向计算
- 清除梯度
- 后向计算
- 权重更新
- 载入权重

不同的神经网络类在神经网络虚基类的基础上继承得到。与基类相比，只添加了不同的构造方法和重载了载入权重的方法。目前定义的网络有 LeNet<sup><a href="#ref1">1</a></sup>、AlexNet<sup><a href="#ref2">2</a></sup>、ResNet18<sup><a href="#ref3">3</a></sup> 和 VGG11<sup><a href="#ref4">4</a></sup>。

<p name = "ref1">[1] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.</p>

<p name = "ref2">[2] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105).</p>

<p name = "ref2">[3] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).</p>

<p name = "ref2">[4] Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.</p>

## 功能模块

功能模块的声明在 `include/tools.h` 中，定义在 `src/tools.cpp` 中，包含四个函数：

```c++
DIR *load_dir(char *);
int load_file(char *, char *, tensor3d<double> *&, int);
tensor1d<double> *idx2onehot(int, int);
double MSE_loss(tensor1d<double> *,
                tensor1d<double> *,
                tensor1d<double> *&);
double CE_loss(int,
               tensor1d<double> *,
               tensor1d<double> *&);
```

依次为：

- 从路径获取文件列表，用于载入数据集
- 从文件中载入张量
- 平均均方差损失函数
- 交叉熵损失函数

# 可扩展性

## 1. 神经网络结构的可扩展

为了便于添加不同的神经网络层或是不同的神经网络结构，本实现中为神经网络层类和神经网络类均添加了虚基类。如此的好处是只要满足虚基类所定义的结构或功能，就可以使用统一的接口进行新结构的添加与整合。

特别地，对于非串行的神经网络结构，如残差块结构、注意力块结构，由于其在推理和训练时的表现与单个神经网络层是类似的，也可以通过本实现中神经网络层的接口声明和定义来实现大的“神经网络层”，从而丰富神经网络种类。

## 2. 数据类型的可扩展

现如今，低浮点精度、低比特的训练和推理已经成为了一个新的方向。为了更好地支持不同数据类型下的推理和训练，本实现中的张量类全部使用模板实现。基于此，可以任意修改神经网络层的数据类型，包括使用自定义的数据类型，而只需要重载自定义类型的相关算符。

# `main.cpp` 功能说明

`main.cpp` 中给出了一个简单的推理示例程序。该程序将载入事先给出的权重和输入特征，进行前向运算。运行后的输出与 `python/forward_val.ipynb` 的运行结果相比较，能够验证程序前向运算的正确性。

由于训练功能耗时太长，在此不给出训练功能的示例程序。

# 运行环境

实现系统环境为 `Ubuntu 18.04.6 LTS`，cmake 版本为 `3.10.2`，g++ 版本为 `7.5.0`。
