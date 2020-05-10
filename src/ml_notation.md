# Suggested Notation for Machine Learning

> This introduces a suggestion of mathematical notation protocol for machine learning.

The field of machine learning is evolving rapidly in recent years. Communication between different researchers and research groups becomes increasingly important. A key challenge for communication arises from inconsistent notation usages among different papers. This proposal suggests a standard for commonly used mathematical notation for machine learning. In this first version, only some notation are mentioned and more notation are left to be done. This proposal will be regularly updated based on the progress of the field. We look forward to more suggestions to improve this proposal in future versions.

## Tabel of Contents

- [Suggested Notation for Machine Learning](#suggested-notation-for-machine-learning)
  - [Tabel of Contents](#tabel-of-contents)
  - [Dataset](#dataset)
  - [Function](#function)
  - [Loss function](#loss-function)
  - [Activation function](#activation-function)
  - [Two-layer neural network](#two-layer-neural-network)
  - [General deep neural network](#general-deep-neural-network)
  - [Complexity](#complexity)
  - [Training](#training)
  - [Fourier Frequency](#fourier-frequency)
  - [Convolution](#convolution)
  - [Notation table](#notation-table)
  - [L-layer neural network](#l-layer-neural-network)
- [Acknowledgements](#acknowledgements)

## Dataset
t
est

Dataset $S=\{\mathbf{z}_i\}^n_{i=1}=\{(\mathbf{x}_i,\mathbf{y}_i)\}^n_{i=1}$ is sampled from a distribution $\mathcal{D}$ over a domain $\mathcal{Z}=\mathcal{X}\times\mathcal{Y}$.

- $\mathcal{X}$ is the instances domain (a set)
- $\mathcal{Y}$ is the label domain (a set)
- $\mathcal{Z}=\mathcal{X}\times\mathcal{Y}$ is the example domain (a set)

Usually, $\mathcal{X}$ is a subset of $\mathbb{R}^d$ and $\mathcal{Y}$ is a subset of $\mathbb{R}^{d_\text{o}}$, where $d$ is the input dimension, $d_\text{o}$ is the ouput dimension.

$n=\#S$ is the number of samples. Wihout specification, $S$ and $n$ are for the training set.

## Function

A hypothesis space is denoted by $\mathcal{H}$. A hypothesis function is denoted by $f_{\mathbf{\theta}}(\mathbf{x})\in\mathcal{H}$ or $f(\mathbf{x};\mathbf{\theta})$ with $f_{\mathbf{\theta}}:\mathcal{X}\to\mathcal{Y}$.

$\mathbf{\theta}$ denotes the set of parameters of $f_{\mathbf{\theta}}$.

If there exists a target function, it is denoted by $f^*$ or $f^*:\mathcal{X}\to\mathcal{Y}$ satisfying $\mathbf{y}_i=f^*(\mathbf{x}_i)$ for $i=1,\dots,n$.

## Loss function

A loss function, denoted by $\ell:\mathcal{H}\times\mathcal{Z}\to\mathbb{R}_{+}:=[0,+\infty)$ measures the difference between a predicted label and a true label, e.g.,

- $L^2$ loss: $\ell(f_{\mathbf{\theta}},\mathbf{z})=(f_{\mathbf{\theta}}(\mathbf{x})-\mathbf{y})^2$, where $\mathbf{z}=(\mathbf{x},\mathbf{y})$. $\ell(f_{\mathbf{\theta}},\mathbf{z})$ can also be written as $\ell(f_{\mathbf{\theta}},\mathbf{y}))$ for convenience.

Empirical risk or training loss for a set $S=\{(\mathbf{x_i},\mathbf{y_i})\}^n_{i=1}$ is denoted by $L_S(\mathbf{\theta})$ or $L_n(\mathbf{\theta})$ or $R_S(\mathbf{\theta})$ or $R_n(\mathbf{\theta})$,

$$
  L_S(\mathbf{\theta})=\frac{1}{n}\sum^n_{i=1}\ell(f_{\mathbf{\theta}}(\mathbf{x}_i),\mathbf{y}_i).
$$
 

The population risk or expected loss is denoted by $L_{\mathcal{D}}$ or $R_{\mathcal{D}}$,

$$
  L_{\mathcal{D}}(\mathbf{\theta})=\mathbb{E}_{\mathcal{D}}\ell(f_{\mathbf{\theta}}(\mathbf{x}),\mathbf{y})),
$$

where $\mathbf{z}=(\mathbf{x},\mathbf{y})$ follows the distribution $\mathcal{D}$.

## Activation function

An activation function is denoted by $\sigma(x)$.

**Example 1**. Some commonly used activation functions are

- $\sigma(x)=\text{ReLU}(x)=\text{max}(0,x)$
- $\sigma(x)=\text{sigmoid}(x)=\dfrac{1}{1+e^{-x}}$
- $\sigma(x)=\tanh(x)$
- $\sigma(x)=\cos x, \sin x$

## Two-layer neural network

The neuron number of the hidden layer is denoted by $m$, The two-layer neural network is

$$
  f_{\mathbf{\theta}}(\mathbf{x})=\sum^m_{j=1}a_j\sigma(\mathbf{w}_j\cdot\mathbf{x}+b_j),
$$

where $\sigma$ is the activation function, $\mathbf{w}_j$ is the input weight, $a_j$ is the output weight, $b_j$ is the bias term. We denote the set of parameters by

$$
  \mathbf{\theta}=(a_1,\ldots,a_m,\mathbf{w}_1,\ldots,\mathbf{w}_m,b_1,\cdots,b_m).
$$

## General deep neural network

The counting of the layer number excludes the input layer. An $L$-layer neural network is denoted by
$$
  f_{\mathbf{\theta}}(\mathbf{x})=\mathbf{W}^{[L-1]}\sigma\circ(\mathbf{W}^{[L-2]})\sigma\circ(\cdots(\mathbf{W}^{[1]}\sigma\circ(\mathbf{W}^{[0]}\mathbf{x}+\mathbf{b}^{[0]})+\mathbf{b}^{[1]})\cdots)+\mathbf{b}^{[L-2]})+\mathbf{b}^{[L-1]},
$$

where $\mathbf{W}^{[l]}\in\mathbb{R}^{m_{l+1}\times m_l}$, $\mathbf{b}^{[l]}=\mathbb{R}^{m_{l+1}}$, $m_0=d_\text{in}=d$, $m_{L}=d_\text{o}$, $\sigma$ is a scalar function and "$\circ$" means entry-wise operation. We denote the set of parameters by

$$
  \mathbf{\theta}=(\mathbf{W}^{[0]},\mathbf{W}^{[1]},\dots,\mathbf{W}^{[L-1]},\mathbf{b}^{[0]},\mathbf{b}^{[1]},\dots,\mathbf{b}^{[L-1]}).
$$

This can also be defined recursively,

$$
  f^{[0]}_{\mathbf{\theta}}(\mathbf{x})=\mathbf{x},
$$
$$
  f^{[l]}_{\mathbf{\theta}}(\mathbf{x})=\sigma\circ(\mathbf{W}^{[l-1]}f^{[l-1]}_{\mathbf{\theta}}(\mathbf{x})+\mathbf{b}^{[l-1]}), \quad 1\le l\le L-1,
$$
$$
  f_{\mathbf{\theta}}(\mathbf{x})=f^{[L]}_{\mathbf{\theta}}(\mathbf{x})=\mathbf{W}^{[L-1]}f^{[L-1]}_{\mathbf{\theta}}(\mathbf{x})+\mathbf{b}^{[L-1]}, \quad 1\le l\le L-1.
$$

## Complexity

The VC-dimension of a hypothesis class $\mathcal{H}$ is denoted as VCdim($\mathcal{H}$).

The Rademacher complexity of a hypothesis space $\mathcal{H}$ on a sample set $S$ is denoted by $R(\mathcal{H}\circ S)$ or $\text{Rad}_S(\mathcal{H})$. The complexity $\text{Rad}_S(\mathcal{H})$ is random because of the randomness of $S$. The expectation of the empirical Rademacher complexity over all samples of size $n$ is denoted by

$$
  \text{Rad}_n(\mathcal{H}) = \mathbb{E}_S\text{Rad}_S(\mathcal{H}).
$$

## Training

The Gradient Descent is oftern denoted by GD. THe Stochastic Gradient Descent is often denoted by SGD.

A batch set is denoted by $B$ and the batch size is denoted by $|B|$.

The learning rate is denoted by $\eta$.

## Fourier Frequency

The discretized frequency is denoted by $\mathbf{k}$, and the continuous frequency is denoted by $\mathbf{\xi}$.

## Convolution

The convolution operation is denoted by $*$.

## Notation table

| symbol                                                                                           | meaning                                               | Latex              | simplied              |
| ------------------------------------------------------------------------------------------------ | ----------------------------------------------------- | ------------------ | --------------------- |
| $\mathbf{x}$                                                                                     | input                                                 | `\bm{x}`           | `\mathbf{x}`          |
| $\mathbf{y}$                                                                                     | output, label                                         | `\bm{y}`           | `\vy`                 |
| $d$                                                                                              | input dimension                                       | `d`                |                       |
| $d_{\text{o}}$                                                                                   | output dimension                                      | `d_{\rm o}`        |                       |
| $n$                                                                                              | number of samples                                     | `n`                |
| $\mathcal{X}$                                                                                    | instances domain (a set)                              | `\mathcal{X}`      | `\fX`                 |
| $\mathcal{Y}$                                                                                    | labels domain (a set)                                 | `\mathcal{Y}`      | `\fY`                 |
| $\mathcal{Z}$                                                                                    | $=\mathcal{X}\times\mathcal{Y}$ example domain        | `\mathcal{Z}`      | `\fZ`                 |
| $\mathcal{H}$                                                                                    | hypothesis space (a set)                              | `\mathcal{H}`      | `\mathcal{H}`         |
| $\mathbf{\theta}$                                                                                | a set of parameters                                   | `\bm{\theta}`      | `\mathbf{\theta}`     |
| $f_{\mathbf{\theta}}: \mathcal{X}\to\mathcal{Y}$                                                 | hypothesis function                                   | `\f_{\bm{\theta}}` | `f_{\mathbf{\theta}}` |
| $f$ or $f^*: \mathcal{X}\to\mathcal{Y}$                                                          | target function                                       | `f, f^*`           |
| $\ell:\mathcal{H}\times \mathcal{Z}\to \mathbb{R}^+$                                             | loss function                                         | `\ell`             |
| $\mathcal{D}$                                                                                    | distribution of $\mathcal{Z}$                         | `\mathcal{D}`      | `\fD`                 |
| $S=\{\mathbf{z}_i\}_{i=1}^n$                                                                     | $=\{(\mathbf{x}_i,\mathbf{y}_i)\}_{i=1}^n$ sample set |
| $L_S(\mathbf{\theta})$, $L_{n}(\mathbf{\theta})$, $R_n(\mathbf{\theta})$, $R_S(\mathbf{\theta})$ | empirical risk or training loss                       |
| $L_D(\mathbf{\theta})$                                                                           | population risk or expected loss                      |
| $\sigma:\mathbb{R}\to\mathbb{R}^+$                                                               | activation function                                   | `\sigma`           |
| $\mathbf{w}_j$                                                                                   | input weight                                          | `\bm{w}_j`         | `\mathbf{w}_j`        |
| $a_j$                                                                                            | output weight                                         | `a_j`              |
| $b_j$                                                                                            | bias term                                             | `b_j`              |
| $f_{\mathbf{\theta}}(\mathbf{x})$ or $f(\mathbf{x};\mathbf{\theta})$                             | neural network                                        | `f_{\bm{\theta}}`  | `f_{\mathbf{\theta}}` |
| $\sum_{j=1}^{m} a_j \sigma (\mathbf{w}_j\cdot \mathbf{x} + b_j)$                                 | two-layer neural network                              |
| $\text{VCdim}(\mathcal{H}$)                                                                      | VC-dimension of $\mathcal{H}$                         |
| $\text{Rad}(\mathcal{H}\circ S)$, $\text{Rad}_{S}(\mathcal{H})$                                  | Rademacher complexity of $\mathcal{H}$ on $S$         |
| ${\rm Rad}_{n} (\mathcal{H})$                                                                    | Rademacher complexity over samples of size $n$        |
| $\text{GD}$                                                                                      | gradient descent                                      |
| $\text{SGD}$                                                                                     | stochastic gradient descent                           |
| $B$                                                                                              | a batch set                                           | `B`                |                       |
| $\vert B\vert$                                                                                   | batch size                                            | `b`                |                       |
| $\eta$                                                                                           | learning rate                                         | `\eta`             |
| $\mathbf{k}$                                                                                     | discretized frequency                                 | `\bm{k}`           | `\mathbf{k}`          |
| $\mathbf{\xi}$                                                                                   | continuous frequency                                  | `\bm{\xi}`         | `\mathbf{xi}`         |  |
| $*$                                                                                              | convolution operation                                 | `*`                |

## L-layer neural network

| symbol                                  | meaning                                                                                                                                 | Latex          | simplied           |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | -------------- | ------------------ |
| $d$                                     | input dimension                                                                                                                         | `d`            |                    |
| $d_{\text{o}}$                          | output dimension                                                                                                                        | `d_{\rm o}`    |                    |
| $m_l$                                   | the number of $l$-th layer neuron, $m_0=d$, $m_{L} = d_{\text{o}}$                                                                      | `m_l`          |
| $\mathbf{W}^{[l]}$                      | the $l$-th layer weight                                                                                                                 | `\bm{W}^{[l]}` | `\mathbf{W}^{[l]}` |
| $\mathbf{b}^{[l]}$                      | the $l$-th layer bias term                                                                                                              | `\bm{b}^{[l]}` | `\mathbf{b}^{[l]}` |
| $\circ$                                 | entry-wise operation                                                                                                                    | `\circ`        |
| $\sigma:\mathbb{R}\to\mathbb{R}^+$      | activation function                                                                                                                     | `\sigma`       |
| $\mathbf{\theta}$                       | $=(\mathbf{W}^{[0]},\ldots,\mathbf{W}^{[L-1]},\mathbf{b}^{[0]},\ldots,\mathbf{b}^{[L-1]})$,  parameters                                 | `\bm{\theta}`  | `\mathbf{\theta}`  |
| $f_{\mathbf{\theta}}^{[0]}(\mathbf{x})$ | $=\mathbf{x}$                                                                                                                           |
| $f_{\mathbf{\theta}}^{[l]}(\mathbf{x})$ | $=\sigma\circ(\mathbf{W}^{[l-1]} f_{\mathbf{\theta}}^{[l-1]}(\mathbf{x}) + \mathbf{b}^{[l-1]})$,  $l$-th  layer output                  |
| $f_{\mathbf{\theta}}(\mathbf{x})$       | $=f_{\mathbf{\theta}}^{[L]}(\mathbf{x})=\mathbf{W}^{[L-1]} f_{\mathbf{\theta}}^{[L-1]}(\mathbf{x}) + \mathbf{b}^{[L-1]}$,  $L$-layer NN |

# Acknowledgements

Chenglong Bao (Tsinghua), Zhengdao Chen (NYU), Bin Dong (Peking), Weinan E (Princeton),  Quanquan Gu (UCLA), Kaizhu Huang (XJTLU), Shi Jin (SJTU), Jian Li (Tsinghua), Lei Li (SJTU), Tiejun Li (Peking),   Zhenguo Li (Huawei), Zhemin Li (NUDT), Shaobo Lin (XJTU), Ziqi Liu (CSRC),  Zichao Long (Peking), Chao Ma (Princeton),  Chao Ma (SJTU), Yuheng Ma (WHU),    Dengyu Meng (XJTU), Wang Miao (Peking),  Pingbing Ming (CAS), Zuoqiang Shi (Tsinghua), Jihong Wang (CSRC), Liwei Wang (Peking), Bican Xia (Peking), Zhouwang Yang (USTC),  Haijun Yu (CAS),  Yang Yuan  (Tsinghua),  Cheng Zhang (Peking),  Lulu Zhang (SJTU), Jiwei Zhang  (WHU),   Pingwen Zhang (Peking), Xiaoqun Zhang (SJTU),  Chengchao Zhao (CSRC), Zhanxing Zhu (Peking), Chuan Zhou (CAS),  Xiang Zhou (cityU). 
