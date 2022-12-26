# 代码说明

由于网络问题，GitHub 始终无法上传，因此将文件上传至如下链接的云盘中，带来不便还望谅解！

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


<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" width="221px" viewBox="-0.5 -0.5 221 111" style="max-width:100%;max-height:111px;"><defs/><g><path d="M 60 55 L 134.45 13.12" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 139.03 10.55 L 134.07 16.01 L 134.45 13.12 L 131.78 11.95 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="all"/><path d="M 60 55 L 133.74 41.17" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 138.9 40.21 L 132.45 43.79 L 133.74 41.17 L 131.59 39.2 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="all"/><path d="M 60 55 L 133.74 68.83" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 138.9 69.79 L 131.59 70.8 L 133.74 68.83 L 132.45 66.21 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="all"/><path d="M 60 55 L 134.45 96.88" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 139.03 99.45 L 131.78 98.05 L 134.45 96.88 L 134.07 93.99 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="all"/><rect x="0" y="40" width="60" height="30" fill="none" stroke="rgb(0, 0, 0)" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe flex-start; width: 50px; height: 1px; padding-top: 55px; margin-left: 6px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: left;"><div style="display: inline-block; font-size: 12px; font-family: &quot;Courier New&quot;; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: all; white-space: normal; overflow-wrap: normal;">tensor</div></div></div></foreignObject><text x="6" y="59" fill="rgb(0, 0, 0)" font-family="Courier New" font-size="12px">tensor</text></switch></g><rect x="140" y="0" width="80" height="20" fill="none" stroke="rgb(0, 0, 0)" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe flex-start; width: 70px; height: 1px; padding-top: 10px; margin-left: 146px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: left;"><div style="display: inline-block; font-size: 12px; font-family: &quot;Courier New&quot;; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: all; white-space: normal; overflow-wrap: normal;">tensor1d</div></div></div></foreignObject><text x="146" y="14" fill="rgb(0, 0, 0)" font-family="Courier New" font-size="12px">tensor1d</text></switch></g><rect x="140" y="30" width="80" height="20" fill="none" stroke="rgb(0, 0, 0)" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe flex-start; width: 70px; height: 1px; padding-top: 40px; margin-left: 146px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: left;"><div style="display: inline-block; font-size: 12px; font-family: &quot;Courier New&quot;; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: all; white-space: normal; overflow-wrap: normal;">tensor2d</div></div></div></foreignObject><text x="146" y="44" fill="rgb(0, 0, 0)" font-family="Courier New" font-size="12px">tensor2d</text></switch></g><rect x="140" y="60" width="80" height="20" fill="none" stroke="rgb(0, 0, 0)" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe flex-start; width: 70px; height: 1px; padding-top: 70px; margin-left: 146px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: left;"><div style="display: inline-block; font-size: 12px; font-family: &quot;Courier New&quot;; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: all; white-space: normal; overflow-wrap: normal;">tensor3d</div></div></div></foreignObject><text x="146" y="74" fill="rgb(0, 0, 0)" font-family="Courier New" font-size="12px">tensor3d</text></switch></g><rect x="140" y="90" width="80" height="20" fill="none" stroke="rgb(0, 0, 0)" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe flex-start; width: 70px; height: 1px; padding-top: 100px; margin-left: 146px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: left;"><div style="display: inline-block; font-size: 12px; font-family: &quot;Courier New&quot;; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: all; white-space: normal; overflow-wrap: normal;">tensor4d</div></div></div></foreignObject><text x="146" y="104" fill="rgb(0, 0, 0)" font-family="Courier New" font-size="12px">tensor4d</text></switch></g></g><switch><g requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"/><a transform="translate(0,-5)" xlink:href="https://www.diagrams.net/doc/faq/svg-export-text-problems" target="_blank"><text text-anchor="middle" font-size="10px" x="50%" y="100%">Text is not SVG - cannot display</text></a></switch></svg>

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

<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" width="362px" viewBox="-0.5 -0.5 362 262" style="max-width:100%;max-height:262px;"><defs/><g><path d="M 60 105 L 106.12 45.05" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 109.32 40.89 L 106.9 47.86 L 106.12 45.05 L 103.2 45.01 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="all"/><path d="M 60 105 L 106.3 169.82" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 109.35 174.09 L 103.38 169.75 L 106.3 169.82 L 107.18 167.04 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="all"/><rect x="0" y="90" width="60" height="30" fill="none" stroke="rgb(0, 0, 0)" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe flex-start; width: 50px; height: 1px; padding-top: 105px; margin-left: 6px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: left;"><div style="display: inline-block; font-size: 12px; font-family: &quot;Courier New&quot;; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: all; white-space: normal; overflow-wrap: normal;">layer</div></div></div></foreignObject><text x="6" y="109" fill="rgb(0, 0, 0)" font-family="Courier New" font-size="12px">layer</text></switch></g><path d="M 180 40 L 234.3 12.85" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 239 10.5 L 233.78 15.72 L 234.3 12.85 L 231.7 11.54 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="all"/><path d="M 180 40 L 233.63 40" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 238.88 40 L 231.88 42.33 L 233.63 40 L 231.88 37.67 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="all"/><path d="M 180 40 L 234.3 67.15" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 239 69.5 L 231.7 68.46 L 234.3 67.15 L 233.78 64.28 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="all"/><rect x="110" y="30" width="70" height="20" fill="none" stroke="rgb(0, 0, 0)" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe flex-start; width: 60px; height: 1px; padding-top: 40px; margin-left: 116px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: left;"><div style="display: inline-block; font-size: 12px; font-family: &quot;Courier New&quot;; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: all; white-space: normal; overflow-wrap: normal;">layer1d</div></div></div></foreignObject><text x="116" y="44" fill="rgb(0, 0, 0)" font-family="Courier New" font-size="12px">layer1d</text></switch></g><path d="M 180 175 L 236.02 104.97" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 239.3 100.87 L 236.75 107.8 L 236.02 104.97 L 233.11 104.88 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="all"/><path d="M 180 175 L 234.91 133.82" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 239.11 130.67 L 234.91 136.74 L 234.91 133.82 L 232.11 133 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="all"/><path d="M 180 175 L 233.82 161.54" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 238.92 160.27 L 232.69 164.23 L 233.82 161.54 L 231.56 159.71 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="all"/><path d="M 180 175 L 233.82 188.46" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 238.92 189.73 L 231.56 190.29 L 233.82 188.46 L 232.69 185.77 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="all"/><path d="M 180 175 L 234.91 216.18" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 239.11 219.33 L 232.11 217 L 234.91 216.18 L 234.91 213.26 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="all"/><path d="M 180 175 L 236.02 245.03" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 239.3 249.13 L 233.11 245.12 L 236.02 245.03 L 236.75 242.2 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="all"/><rect x="110" y="160" width="70" height="30" fill="none" stroke="rgb(0, 0, 0)" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe flex-start; width: 60px; height: 1px; padding-top: 175px; margin-left: 116px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: left;"><div style="display: inline-block; font-size: 12px; font-family: &quot;Courier New&quot;; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: all; white-space: normal; overflow-wrap: normal;">layer3d</div></div></div></foreignObject><text x="116" y="179" fill="rgb(0, 0, 0)" font-family="Courier New" font-size="12px">layer3d</text></switch></g><rect x="240" y="0" width="120" height="20" fill="none" stroke="rgb(0, 0, 0)" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe flex-start; width: 110px; height: 1px; padding-top: 10px; margin-left: 246px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: left;"><div style="display: inline-block; font-size: 12px; font-family: &quot;Courier New&quot;; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: all; white-space: normal; overflow-wrap: normal;">layer_fc</div></div></div></foreignObject><text x="246" y="14" fill="rgb(0, 0, 0)" font-family="Courier New" font-size="12px">layer_fc</text></switch></g><rect x="240" y="30" width="120" height="20" fill="none" stroke="rgb(0, 0, 0)" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe flex-start; width: 110px; height: 1px; padding-top: 40px; margin-left: 246px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: left;"><div style="display: inline-block; font-size: 12px; font-family: &quot;Courier New&quot;; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: all; white-space: normal; overflow-wrap: normal;">layer_ReLU1d</div></div></div></foreignObject><text x="246" y="44" fill="rgb(0, 0, 0)" font-family="Courier New" font-size="12px">layer_ReLU1d</text></switch></g><rect x="240" y="60" width="120" height="20" fill="none" stroke="rgb(0, 0, 0)" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe flex-start; width: 110px; height: 1px; padding-top: 70px; margin-left: 246px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: left;"><div style="display: inline-block; font-size: 12px; font-family: &quot;Courier New&quot;; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: all; white-space: normal; overflow-wrap: normal;">layer_sigmoid1d</div></div></div></foreignObject><text x="246" y="74" fill="rgb(0, 0, 0)" font-family="Courier New" font-size="12px">layer_sigmoid1d</text></switch></g><rect x="240" y="90" width="120" height="20" fill="none" stroke="rgb(0, 0, 0)" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe flex-start; width: 110px; height: 1px; padding-top: 100px; margin-left: 246px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: left;"><div style="display: inline-block; font-size: 12px; font-family: &quot;Courier New&quot;; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: all; white-space: normal; overflow-wrap: normal;">layer_conv</div></div></div></foreignObject><text x="246" y="104" fill="rgb(0, 0, 0)" font-family="Courier New" font-size="12px">layer_conv</text></switch></g><rect x="240" y="120" width="120" height="20" fill="none" stroke="rgb(0, 0, 0)" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe flex-start; width: 110px; height: 1px; padding-top: 130px; margin-left: 246px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: left;"><div style="display: inline-block; font-size: 12px; font-family: &quot;Courier New&quot;; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: all; white-space: normal; overflow-wrap: normal;">layer_ReLU3d</div></div></div></foreignObject><text x="246" y="134" fill="rgb(0, 0, 0)" font-family="Courier New" font-size="12px">layer_ReLU3d</text></switch></g><rect x="240" y="150" width="120" height="20" fill="none" stroke="rgb(0, 0, 0)" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe flex-start; width: 110px; height: 1px; padding-top: 160px; margin-left: 246px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: left;"><div style="display: inline-block; font-size: 12px; font-family: &quot;Courier New&quot;; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: all; white-space: normal; overflow-wrap: normal;">layer_sigmoid3d</div></div></div></foreignObject><text x="246" y="164" fill="rgb(0, 0, 0)" font-family="Courier New" font-size="12px">layer_sigmoid3d</text></switch></g><rect x="240" y="180" width="120" height="20" fill="none" stroke="rgb(0, 0, 0)" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe flex-start; width: 110px; height: 1px; padding-top: 190px; margin-left: 246px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: left;"><div style="display: inline-block; font-size: 12px; font-family: &quot;Courier New&quot;; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: all; white-space: normal; overflow-wrap: normal;">layer_avgpool</div></div></div></foreignObject><text x="246" y="194" fill="rgb(0, 0, 0)" font-family="Courier New" font-size="12px">layer_avgpool</text></switch></g><rect x="240" y="210" width="120" height="20" fill="none" stroke="rgb(0, 0, 0)" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe flex-start; width: 110px; height: 1px; padding-top: 220px; margin-left: 246px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: left;"><div style="display: inline-block; font-size: 12px; font-family: &quot;Courier New&quot;; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: all; white-space: normal; overflow-wrap: normal;">layer_maxpool</div></div></div></foreignObject><text x="246" y="224" fill="rgb(0, 0, 0)" font-family="Courier New" font-size="12px">layer_maxpool</text></switch></g><rect x="240" y="240" width="120" height="20" fill="none" stroke="rgb(0, 0, 0)" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe flex-start; width: 110px; height: 1px; padding-top: 250px; margin-left: 246px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: left;"><div style="display: inline-block; font-size: 12px; font-family: &quot;Courier New&quot;; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: all; white-space: normal; overflow-wrap: normal;">layer_res</div></div></div></foreignObject><text x="246" y="254" fill="rgb(0, 0, 0)" font-family="Courier New" font-size="12px">layer_res</text></switch></g></g><switch><g requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"/><a transform="translate(0,-5)" xlink:href="https://www.diagrams.net/doc/faq/svg-export-text-problems" target="_blank"><text text-anchor="middle" font-size="10px" x="50%" y="100%">Text is not SVG - cannot display</text></a></switch></svg>

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

<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" width="221px" viewBox="-0.5 -0.5 221 111" style="max-width:100%;max-height:111px;"><defs/><g><path d="M 60 55 L 134.45 13.12" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 139.03 10.55 L 134.07 16.01 L 134.45 13.12 L 131.78 11.95 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="all"/><path d="M 60 55 L 133.74 41.17" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 138.9 40.21 L 132.45 43.79 L 133.74 41.17 L 131.59 39.2 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="all"/><path d="M 60 55 L 133.74 68.83" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 138.9 69.79 L 131.59 70.8 L 133.74 68.83 L 132.45 66.21 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="all"/><path d="M 60 55 L 134.45 96.88" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 139.03 99.45 L 131.78 98.05 L 134.45 96.88 L 134.07 93.99 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="all"/><rect x="0" y="40" width="60" height="30" fill="none" stroke="rgb(0, 0, 0)" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe flex-start; width: 50px; height: 1px; padding-top: 55px; margin-left: 6px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: left;"><div style="display: inline-block; font-size: 12px; font-family: &quot;Courier New&quot;; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: all; white-space: normal; overflow-wrap: normal;">network</div></div></div></foreignObject><text x="6" y="59" fill="rgb(0, 0, 0)" font-family="Courier New" font-size="12px">network</text></switch></g><rect x="140" y="0" width="80" height="20" fill="none" stroke="rgb(0, 0, 0)" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe flex-start; width: 70px; height: 1px; padding-top: 10px; margin-left: 146px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: left;"><div style="display: inline-block; font-size: 12px; font-family: &quot;Courier New&quot;; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: all; white-space: normal; overflow-wrap: normal;">LeNet</div></div></div></foreignObject><text x="146" y="14" fill="rgb(0, 0, 0)" font-family="Courier New" font-size="12px">LeNet</text></switch></g><rect x="140" y="30" width="80" height="20" fill="none" stroke="rgb(0, 0, 0)" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe flex-start; width: 70px; height: 1px; padding-top: 40px; margin-left: 146px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: left;"><div style="display: inline-block; font-size: 12px; font-family: &quot;Courier New&quot;; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: all; white-space: normal; overflow-wrap: normal;">AlexNet</div></div></div></foreignObject><text x="146" y="44" fill="rgb(0, 0, 0)" font-family="Courier New" font-size="12px">AlexNet</text></switch></g><rect x="140" y="60" width="80" height="20" fill="none" stroke="rgb(0, 0, 0)" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe flex-start; width: 70px; height: 1px; padding-top: 70px; margin-left: 146px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: left;"><div style="display: inline-block; font-size: 12px; font-family: &quot;Courier New&quot;; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: all; white-space: normal; overflow-wrap: normal;">ResNet18</div></div></div></foreignObject><text x="146" y="74" fill="rgb(0, 0, 0)" font-family="Courier New" font-size="12px">ResNet18</text></switch></g><rect x="140" y="90" width="80" height="20" fill="none" stroke="rgb(0, 0, 0)" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe flex-start; width: 70px; height: 1px; padding-top: 100px; margin-left: 146px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: left;"><div style="display: inline-block; font-size: 12px; font-family: &quot;Courier New&quot;; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: all; white-space: normal; overflow-wrap: normal;">VGG11</div></div></div></foreignObject><text x="146" y="104" fill="rgb(0, 0, 0)" font-family="Courier New" font-size="12px">VGG11</text></switch></g></g><switch><g requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"/><a transform="translate(0,-5)" xlink:href="https://www.diagrams.net/doc/faq/svg-export-text-problems" target="_blank"><text text-anchor="middle" font-size="10px" x="50%" y="100%">Text is not SVG - cannot display</text></a></switch></svg>

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
