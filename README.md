# Suggested Notation for Machine Learning

## Authors

- **Beijing Academy of Artificial Intelligence (北京智源人工智能研究院)**
- **Peking University (北京大学)**
- **Shanghai Jiao Tong University (上海交通大学)**
- **[Zhi-qin John Xu (许志钦)](mailto:xuzhiqin@sjtu.edu.cn), [Tao Luo (罗涛)](mailto:luo196@purdue.edu), [Zheng Ma (马征)](mailto:ma531@purdue.edu), [Yaoyu Zhang (张耀宇)](mailto:yaoyu@ias.edu)** - _Initial work_

## Introduction

> This introduces a suggestion of mathematical notation protocol for machine learning.

The field of machine learning is evolving rapidly in recent years. Communication between different researchers and research groups becomes increasingly important. A key challenge for communication arises from inconsistent notation usages among different papers. This proposal suggests a standard for commonly used mathematical notation for machine learning. In this first version, only some notation are mentioned and more notation are left to be done. This proposal will be regularly updated based on the progress of the field. We look forward to more suggestions to improve this proposal in future versions.

## Tabel of Contents

- [Suggested Notation for Machine Learning](#suggested-notation-for-machine-learning)
  - [Authors](#authors)
  - [Introduction](#introduction)
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

Dataset <img alt="$S=\{\mathbf{z}_i\}^n_{i=1}=\{(\mathbf{x}_i,\mathbf{y}_i)\}^n_{i=1}$" src="svgs/a62b8a89e44f4588d548edf23e5964bd.svg" align="middle" width="196.01672969999998pt" height="24.65753399999998pt"/> is sampled from a distribution <img alt="$\mathcal{D}$" src="svgs/eaf85f2b753a4c7585def4cc7ecade43.svg" align="middle" width="13.13706569999999pt" height="22.465723500000017pt"/> over a domain <img alt="$\mathcal{Z}=\mathcal{X}\times\mathcal{Y}$" src="svgs/7b416174c3d0e087d28a4cc81bae17fd.svg" align="middle" width="81.69842999999999pt" height="22.465723500000017pt"/>.

- <img alt="$\mathcal{X}$" src="svgs/7da75f4e61cdeabf944740206b511812.svg" align="middle" width="14.132466149999988pt" height="22.465723500000017pt"/> is the instances domain (a set)
- <img alt="$\mathcal{Y}$" src="svgs/fce9019a5e1fa63e079199cd9b11c55e.svg" align="middle" width="12.337954199999992pt" height="22.465723500000017pt"/> is the label domain (a set)
- <img alt="$\mathcal{Z}=\mathcal{X}\times\mathcal{Y}$" src="svgs/7b416174c3d0e087d28a4cc81bae17fd.svg" align="middle" width="81.69842999999999pt" height="22.465723500000017pt"/> is the example domain (a set)

Usually, <img alt="$\mathcal{X}$" src="svgs/7da75f4e61cdeabf944740206b511812.svg" align="middle" width="14.132466149999988pt" height="22.465723500000017pt"/> is a subset of <img alt="$\mathbb{R}^d$" src="svgs/435f1061aa6f25938c3c3515c083d06c.svg" align="middle" width="18.71525699999999pt" height="27.91243950000002pt"/> and <img alt="$\mathcal{Y}$" src="svgs/fce9019a5e1fa63e079199cd9b11c55e.svg" align="middle" width="12.337954199999992pt" height="22.465723500000017pt"/> is a subset of <img alt="$\mathbb{R}^{d_\text{o}}$" src="svgs/8cf8e83175c24764c15718de79a77a04.svg" align="middle" width="24.308956649999992pt" height="27.91243950000002pt"/>, where <img alt="$d$" src="svgs/2103f85b8b1477f430fc407cad462224.svg" align="middle" width="8.55596444999999pt" height="22.831056599999986pt"/> is the input dimension, <img alt="$d_\text{o}$" src="svgs/d36df281d4cb7796ad889d761f50712c.svg" align="middle" width="15.10851044999999pt" height="22.831056599999986pt"/> is the ouput dimension.

<img alt="$n=\#S$" src="svgs/62bf1fa8e6d3a545fb832bd073c542dd.svg" align="middle" width="56.510581049999985pt" height="22.831056599999986pt"/> is the number of samples. Wihout specification, <img alt="$S$" src="svgs/e257acd1ccbe7fcb654708f1a866bfe9.svg" align="middle" width="11.027402099999989pt" height="22.465723500000017pt"/> and <img alt="$n$" src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg" align="middle" width="9.86687624999999pt" height="14.15524440000002pt"/> are for the training set.

## Function

A hypothesis space is denoted by <img alt="$\mathcal{H}$" src="svgs/8209c0f8b3c5233ea2e20dae55588c43.svg" align="middle" width="14.041179899999989pt" height="22.465723500000017pt"/>. A hypothesis function is denoted by <img alt="$f_{\mathbf{\theta}}(\mathbf{x})\in\mathcal{H}$" src="svgs/21516e47cad2991185d50716b54b7ccd.svg" align="middle" width="72.38002035pt" height="24.65753399999998pt"/> or <img alt="$f(\mathbf{x};\mathbf{\theta})$" src="svgs/cf2dd6c8d8fd7e223ded07a2d09f5ff4.svg" align="middle" width="48.05937344999999pt" height="24.65753399999998pt"/> with <img alt="$f_{\mathbf{\theta}}:\mathcal{X}\to\mathcal{Y}$" src="svgs/cddc713171a7d82a6a94778d48fa9dad.svg" align="middle" width="81.22459289999999pt" height="22.831056599999986pt"/>.

<img alt="$\mathbf{\theta}$" src="svgs/6fccf0465699020081a15631f4a45ae1.svg" align="middle" width="8.17352744999999pt" height="22.831056599999986pt"/> denotes the set of parameters of <img alt="$f_{\mathbf{\theta}}$" src="svgs/43263b5b62e73bb17cf793dc765b7083.svg" align="middle" width="14.66328269999999pt" height="22.831056599999986pt"/>.

If there exists a target function, it is denoted by <img alt="$f^*$" src="svgs/baa7cc2f36af87229745595791fda227.svg" align="middle" width="16.55260694999999pt" height="22.831056599999986pt"/> or <img alt="$f^*:\mathcal{X}\to\mathcal{Y}$" src="svgs/78d6c299386fc79de759e7449c5fba27.svg" align="middle" width="83.11394024999998pt" height="22.831056599999986pt"/> satisfying <img alt="$\mathbf{y}_i=f^*(\mathbf{x}_i)$" src="svgs/4d877e4e6f3a8a26978f0124709d5f9a.svg" align="middle" width="82.97739119999999pt" height="24.65753399999998pt"/> for <img alt="$i=1,\dots,n$" src="svgs/ae697d8a49bffcb50cc01bc8a09826f7.svg" align="middle" width="82.19635874999999pt" height="21.68300969999999pt"/>.

## Loss function

A loss function, denoted by <img alt="$\ell:\mathcal{H}\times\mathcal{Z}\to\mathbb{R}_{+}:=[0,+\infty)$" src="svgs/91248c7c221d6ed56762761f1038d39f.svg" align="middle" width="198.44712029999997pt" height="24.65753399999998pt"/> measures the difference between a predicted label and a true label, e.g.,

- <img alt="$L^2$" src="svgs/e8831293b846e3a3799cd6a02e4a0cd9.svg" align="middle" width="17.73978854999999pt" height="26.76175259999998pt"/> loss: <img alt="$\ell(f_{\mathbf{\theta}},\mathbf{z})=(f_{\mathbf{\theta}}(\mathbf{x})-\mathbf{y})^2$" src="svgs/267cbf28f1c2d7238eeb97e3f0c38b68.svg" align="middle" width="160.66181009999997pt" height="26.76175259999998pt"/>, where <img alt="$\mathbf{z}=(\mathbf{x},\mathbf{y})$" src="svgs/02e9c08c82c033896f32f3bf6b2ebb59.svg" align="middle" width="70.62752894999998pt" height="24.65753399999998pt"/>. <img alt="$\ell(f_{\mathbf{\theta}},\mathbf{z})$" src="svgs/d59371cab861973036670c707757eb37.svg" align="middle" width="50.82761969999999pt" height="24.65753399999998pt"/> can also be written as <img alt="$\ell(f_{\mathbf{\theta}},\mathbf{y}))$" src="svgs/2c7e0a944f5282a0a8ed736c8c2d32ad.svg" align="middle" width="59.05823879999999pt" height="24.65753399999998pt"/> for convenience.

Empirical risk or training loss for a set <img alt="$S=\{(\mathbf{x_i},\mathbf{y_i})\}^n_{i=1}$" src="svgs/4509212c364bba86adda3e6c197be90c.svg" align="middle" width="120.72371684999999pt" height="24.65753399999998pt"/> is denoted by <img alt="$L_S(\mathbf{\theta})$" src="svgs/800fde4ef8c4cf09b319be12c03a1d50.svg" align="middle" width="41.66906699999999pt" height="24.65753399999998pt"/> or <img alt="$L_n(\mathbf{\theta})$" src="svgs/5b927af53cdeff9f6707e49201987ae0.svg" align="middle" width="41.094140999999986pt" height="24.65753399999998pt"/> or <img alt="$R_S(\mathbf{\theta})$" src="svgs/51ac91bc610ae6bd004d35c9c484fa51.svg" align="middle" width="42.96329894999999pt" height="24.65753399999998pt"/> or <img alt="$R_n(\mathbf{\theta})$" src="svgs/0d8570bb69d1594ab193d13900d33570.svg" align="middle" width="42.38837294999998pt" height="24.65753399999998pt"/>,

<p align="center"><img alt="$$&#10;  L_S(\mathbf{\theta})=\frac{1}{n}\sum^n_{i=1}\ell(f_{\mathbf{\theta}}(\mathbf{x}_i),\mathbf{y}_i).&#10;$$" src="svgs/b84cae2647caeb5068108f22f3a902c2.svg" align="middle" width="197.29974825pt" height="44.89738935pt"/></p>

The population risk or expected loss is denoted by <img alt="$L_{\mathcal{D}}$" src="svgs/f12e3d7a8c582957ddc54ac589792f7f.svg" align="middle" width="21.77590469999999pt" height="22.465723500000017pt"/> or <img alt="$R_{\mathcal{D}}$" src="svgs/7b56ba3aa6d3733b96b42edc4307dd7d.svg" align="middle" width="23.070136649999988pt" height="22.465723500000017pt"/>,

<p align="center"><img alt="$$&#10;  L_{\mathcal{D}}(\mathbf{\theta})=\mathbb{E}_{\mathcal{D}}\ell(f_{\mathbf{\theta}}(\mathbf{x}),\mathbf{y})),&#10;$$" src="svgs/c3b0bdd54fe608c69bd26ca6b16aebe2.svg" align="middle" width="174.2308986pt" height="16.438356pt"/></p>

where <img alt="$\mathbf{z}=(\mathbf{x},\mathbf{y})$" src="svgs/02e9c08c82c033896f32f3bf6b2ebb59.svg" align="middle" width="70.62752894999998pt" height="24.65753399999998pt"/> follows the distribution <img alt="$\mathcal{D}$" src="svgs/eaf85f2b753a4c7585def4cc7ecade43.svg" align="middle" width="13.13706569999999pt" height="22.465723500000017pt"/>.

## Activation function

An activation function is denoted by <img alt="$\sigma(x)$" src="svgs/b9b27f3deff0db82f962a8505706e620.svg" align="middle" width="32.16330314999999pt" height="24.65753399999998pt"/>.

**Example 1**. Some commonly used activation functions are

- <img alt="$\sigma(x)=\text{ReLU}(x)=\text{max}(0,x)$" src="svgs/d215872452b1ebb225b1e576edcb7879.svg" align="middle" width="208.48739834999995pt" height="24.65753399999998pt"/>
- <img alt="$\sigma(x)=\text{sigmoid}(x)=\dfrac{1}{1+e^{-x}}$" src="svgs/a4d649352a659f0b7e0cc3b5b5670bd3.svg" align="middle" width="209.55236114999997pt" height="43.42856099999997pt"/>
- <img alt="$\sigma(x)=\tanh(x)$" src="svgs/f64afc209f5ec2703004ba8141966a8f.svg" align="middle" width="109.13816759999997pt" height="24.65753399999998pt"/>
- <img alt="$\sigma(x)=\cos x, \sin x$" src="svgs/0b557fdfdec1313c966fb5c9cfbee5a1.svg" align="middle" width="127.84794989999999pt" height="24.65753399999998pt"/>

## Two-layer neural network

The neuron number of the hidden layer is denoted by <img alt="$m$" src="svgs/0e51a2dede42189d77627c4d742822c3.svg" align="middle" width="14.433101099999991pt" height="14.15524440000002pt"/>, The two-layer neural network is

<p align="center"><img alt="$$&#10;  f_{\mathbf{\theta}}(\mathbf{x})=\sum^m_{j=1}a_j\sigma(\mathbf{w}_j\cdot\mathbf{x}+b_j),&#10;$$" src="svgs/bcbf19b6c41d3d362d805920319da7e6.svg" align="middle" width="206.10021794999997pt" height="47.1348339pt"/></p>

where <img alt="$\sigma$" src="svgs/8cda31ed38c6d59d14ebefa440099572.svg" align="middle" width="9.98290094999999pt" height="14.15524440000002pt"/> is the activation function, <img alt="$\mathbf{w}_j$" src="svgs/831047ac6f850b0d588c94d84fc6f4c1.svg" align="middle" width="19.75740524999999pt" height="14.611878600000017pt"/> is the input weight, <img alt="$a_j$" src="svgs/3fd897df5707a411645a54460183e3cd.svg" align="middle" width="14.793662399999992pt" height="14.15524440000002pt"/> is the output weight, <img alt="$b_j$" src="svgs/2020a79c00e140ee1a054ecab57a289c.svg" align="middle" width="13.15930604999999pt" height="22.831056599999986pt"/> is the bias term. We denote the set of parameters by

<p align="center"><img alt="$$&#10;  \mathbf{\theta}=(a_1,\ldots,a_m,\mathbf{w}_1,\ldots,\mathbf{w}_m,b_1,\cdots,b_m).&#10;$$" src="svgs/aa9b11397a4c6643cb07e6b991c9fc7e.svg" align="middle" width="292.7598531pt" height="16.438356pt"/></p>

## General deep neural network

The counting of the layer number excludes the input layer. An <img alt="$L$" src="svgs/ddcb483302ed36a59286424aa5e0be17.svg" align="middle" width="11.18724254999999pt" height="22.465723500000017pt"/>-layer neural network is denoted by

<p align="center"><img alt="$$&#10;  f_{\mathbf{\theta}}(\mathbf{x})=\mathbf{W}^{[L-1]}\sigma\circ(\mathbf{W}^{[L-2]})\sigma\circ(\cdots(\mathbf{W}^{[1]}\sigma\circ(\mathbf{W}^{[0]}\mathbf{x}+\mathbf{b}^{[0]})+\mathbf{b}^{[1]})\cdots)+\mathbf{b}^{[L-2]})+\mathbf{b}^{[L-1]},&#10;$$" src="svgs/26e99c0533a05446e1e6ce80dfaecfaa.svg" align="middle" width="652.6473376499999pt" height="19.526994300000002pt"/></p>

where <img alt="$\mathbf{W}^{[l]}\in\mathbb{R}^{m_{l+1}\times m_l}$" src="svgs/865e70ab5839636feaab8a6745125c4a.svg" align="middle" width="120.62059019999998pt" height="29.190975000000005pt"/>, <img alt="$\mathbf{b}^{[l]}=\mathbb{R}^{m_{l+1}}$" src="svgs/998139a600e0e2203867005393bb05b4.svg" align="middle" width="86.43477479999999pt" height="29.190975000000005pt"/>, <img alt="$m_0=d_\text{in}=d$" src="svgs/b5142f01744a994ace1bc28b20b87eed.svg" align="middle" width="94.55845409999999pt" height="22.831056599999986pt"/>, <img alt="$m_{L}=d_\text{o}$" src="svgs/366a5d477d66fb8c7c0d12af876c22e3.svg" align="middle" width="61.29947999999999pt" height="22.831056599999986pt"/>, <img alt="$\sigma$" src="svgs/8cda31ed38c6d59d14ebefa440099572.svg" align="middle" width="9.98290094999999pt" height="14.15524440000002pt"/> is a scalar function and "<img alt="$\circ$" src="svgs/c0463eeb4772bfde779c20d52901d01b.svg" align="middle" width="8.219209349999991pt" height="14.611911599999981pt"/>" means entry-wise operation. We denote the set of parameters by

<p align="center"><img alt="$$&#10;  \mathbf{\theta}=(\mathbf{W}^{[0]},\mathbf{W}^{[1]},\dots,\mathbf{W}^{[L-1]},\mathbf{b}^{[0]},\mathbf{b}^{[1]},\dots,\mathbf{b}^{[L-1]}).&#10;$$" src="svgs/4666610504d6d382d12463f8941f507c.svg" align="middle" width="360.8326854pt" height="19.526994300000002pt"/></p>

This can also be defined recursively,

<p align="center"><img alt="$$&#10;  f^{[0]}_{\mathbf{\theta}}(\mathbf{x})=\mathbf{x},&#10;$$" src="svgs/60638ec6e9895b57f49ac860aa8d7c0f.svg" align="middle" width="83.85835424999999pt" height="22.127694599999998pt"/></p>

<p align="center"><img alt="$$&#10;  f^{[l]}_{\mathbf{\theta}}(\mathbf{x})=\sigma\circ(\mathbf{W}^{[l-1]}f^{[l-1]}_{\mathbf{\theta}}(\mathbf{x})+\mathbf{b}^{[l-1]}), \quad 1\le l\le L-1,&#10;$$" src="svgs/c6ed8c3c10c3702b34c8350acc55b0b8.svg" align="middle" width="401.2952526pt" height="22.127694599999998pt"/></p>

<p align="center"><img alt="$$&#10;  f_{\mathbf{\theta}}(\mathbf{x})=f^{[L]}_{\mathbf{\theta}}(\mathbf{x})=\mathbf{W}^{[L-1]}f^{[L-1]}_{\mathbf{\theta}}(\mathbf{x})+\mathbf{b}^{[L-1]}, \quad 1\le l\le L-1.&#10;$$" src="svgs/668c8dfd4833fded1b6f21375b5454d0.svg" align="middle" width="442.34552009999993pt" height="22.127694599999998pt"/></p>

## Complexity

The VC-dimension of a hypothesis class <img alt="$\mathcal{H}$" src="svgs/8209c0f8b3c5233ea2e20dae55588c43.svg" align="middle" width="14.041179899999989pt" height="22.465723500000017pt"/> is denoted as VCdim(<img alt="$\mathcal{H}$" src="svgs/8209c0f8b3c5233ea2e20dae55588c43.svg" align="middle" width="14.041179899999989pt" height="22.465723500000017pt"/>).

The Rademacher complexity of a hypothesis space <img alt="$\mathcal{H}$" src="svgs/8209c0f8b3c5233ea2e20dae55588c43.svg" align="middle" width="14.041179899999989pt" height="22.465723500000017pt"/> on a sample set <img alt="$S$" src="svgs/e257acd1ccbe7fcb654708f1a866bfe9.svg" align="middle" width="11.027402099999989pt" height="22.465723500000017pt"/> is denoted by <img alt="$R(\mathcal{H}\circ S)$" src="svgs/eb048e4d3123034dac2256effd67ad18.svg" align="middle" width="65.98741049999998pt" height="24.65753399999998pt"/> or <img alt="$\text{Rad}_S(\mathcal{H})$" src="svgs/aba734a98a13ed03fa63549be906337b.svg" align="middle" width="65.80156769999999pt" height="24.65753399999998pt"/>. The complexity <img alt="$\text{Rad}_S(\mathcal{H})$" src="svgs/aba734a98a13ed03fa63549be906337b.svg" align="middle" width="65.80156769999999pt" height="24.65753399999998pt"/> is random because of the randomness of <img alt="$S$" src="svgs/e257acd1ccbe7fcb654708f1a866bfe9.svg" align="middle" width="11.027402099999989pt" height="22.465723500000017pt"/>. The expectation of the empirical Rademacher complexity over all samples of size <img alt="$n$" src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg" align="middle" width="9.86687624999999pt" height="14.15524440000002pt"/> is denoted by

<p align="center"><img alt="$$&#10;  \text{Rad}_n(\mathcal{H}) = \mathbb{E}_S\text{Rad}_S(\mathcal{H}).&#10;$$" src="svgs/6a8c078cff5f892c588c9f5baeb0c908.svg" align="middle" width="177.9938688pt" height="16.438356pt"/></p>

## Training

The Gradient Descent is often denoted by GD. The Stochastic Gradient Descent is often denoted by SGD.

A batch set is denoted by <img alt="$B$" src="svgs/61e84f854bc6258d4108d08d4c4a0852.svg" align="middle" width="13.29340979999999pt" height="22.465723500000017pt"/> and the batch size is denoted by <img alt="$|B|$" src="svgs/007b57eceda75cfb83dcf22bd67fada1.svg" align="middle" width="22.42585124999999pt" height="24.65753399999998pt"/>.

The learning rate is denoted by <img alt="$\eta$" src="svgs/1d0496971a2775f4887d1df25cea4f7e.svg" align="middle" width="8.751954749999989pt" height="14.15524440000002pt"/>.

## Fourier Frequency

The discretized frequency is denoted by <img alt="$\mathbf{k}$" src="svgs/9d152c065089da4147fb86e392670ac8.svg" align="middle" width="9.97711604999999pt" height="22.831056599999986pt"/>, and the continuous frequency is denoted by <img alt="$\mathbf{\xi}$" src="svgs/8107e598d348f99cde7fc8b59875b9b0.svg" align="middle" width="7.94809454999999pt" height="22.831056599999986pt"/>.

## Convolution

The convolution operation is denoted by <img alt="$*$" src="svgs/7c74eeb32158ff7c4f67d191b95450fb.svg" align="middle" width="8.219209349999991pt" height="15.296829900000011pt"/>.

## Notation table

| symbol                                                                                           | meaning                                               | Latex              | simplied              |
| ------------------------------------------------------------------------------------------------ | ----------------------------------------------------- | ------------------ | --------------------- |
| <img alt="$\mathbf{x}$" src="svgs/b0ea07dc5c00127344a1cad40467b8de.svg" align="middle" width="9.97711604999999pt" height="14.611878600000017pt"/>                                                                                     | input                                                 | `\bm{x}`           | `\mathbf{x}`          |
| <img alt="$\mathbf{y}$" src="svgs/1da18d2de6d16a18e780cd6c435a2936.svg" align="middle" width="10.239687149999991pt" height="14.611878600000017pt"/>                                                                                     | output, label                                         | `\bm{y}`           | `\vy`                 |
| <img alt="$d$" src="svgs/2103f85b8b1477f430fc407cad462224.svg" align="middle" width="8.55596444999999pt" height="22.831056599999986pt"/>                                                                                              | input dimension                                       | `d`                |                       |
| <img alt="$d_{\text{o}}$" src="svgs/8f865e6069b66d7a0e8a1f6f72400f41.svg" align="middle" width="15.10851044999999pt" height="22.831056599999986pt"/>                                                                                   | output dimension                                      | `d_{\rm o}`        |                       |
| <img alt="$n$" src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg" align="middle" width="9.86687624999999pt" height="14.15524440000002pt"/>                                                                                              | number of samples                                     | `n`                |
| <img alt="$\mathcal{X}$" src="svgs/7da75f4e61cdeabf944740206b511812.svg" align="middle" width="14.132466149999988pt" height="22.465723500000017pt"/>                                                                                    | instances domain (a set)                              | `\mathcal{X}`      | `\fX`                 |
| <img alt="$\mathcal{Y}$" src="svgs/fce9019a5e1fa63e079199cd9b11c55e.svg" align="middle" width="12.337954199999992pt" height="22.465723500000017pt"/>                                                                                    | labels domain (a set)                                 | `\mathcal{Y}`      | `\fY`                 |
| <img alt="$\mathcal{Z}$" src="svgs/a51b274f1c4520f1cc0b3ee59d7a38e1.svg" align="middle" width="13.21921094999999pt" height="22.465723500000017pt"/>                                                                                    | <img alt="$=\mathcal{X}\times\mathcal{Y}$" src="svgs/01c5b24158b66a590fee8a779d98c7b6.svg" align="middle" width="63.913139399999984pt" height="22.465723500000017pt"/> example domain        | `\mathcal{Z}`      | `\fZ`                 |
| <img alt="$\mathcal{H}$" src="svgs/8209c0f8b3c5233ea2e20dae55588c43.svg" align="middle" width="14.041179899999989pt" height="22.465723500000017pt"/>                                                                                    | hypothesis space (a set)                              | `\mathcal{H}`      | `\mathcal{H}`         |
| <img alt="$\mathbf{\theta}$" src="svgs/6fccf0465699020081a15631f4a45ae1.svg" align="middle" width="8.17352744999999pt" height="22.831056599999986pt"/>                                                                                | a set of parameters                                   | `\bm{\theta}`      | `\mathbf{\theta}`     |
| <img alt="$f_{\mathbf{\theta}}: \mathcal{X}\to\mathcal{Y}$" src="svgs/09eb4790c37751288e62ebbc4a99190d.svg" align="middle" width="81.22459289999999pt" height="22.831056599999986pt"/>                                                 | hypothesis function                                   | `\f_{\bm{\theta}}` | `f_{\mathbf{\theta}}` |
| <img alt="$f$" src="svgs/190083ef7a1625fbc75f243cffb9c96d.svg" align="middle" width="9.81741584999999pt" height="22.831056599999986pt"/> or <img alt="$f^*: \mathcal{X}\to\mathcal{Y}$" src="svgs/a48536501c8099f27b17f0147fbb43a0.svg" align="middle" width="83.11394024999998pt" height="22.831056599999986pt"/>                                                          | target function                                       | `f, f^*`           |
| <img alt="$\ell:\mathcal{H}\times \mathcal{Z}\to \mathbb{R}^+$" src="svgs/64e4c6b981b5ba045a299994ba55aabb.svg" align="middle" width="115.43348354999998pt" height="26.17730939999998pt"/>                                             | loss function                                         | `\ell`             |
| <img alt="$\mathcal{D}$" src="svgs/eaf85f2b753a4c7585def4cc7ecade43.svg" align="middle" width="13.13706569999999pt" height="22.465723500000017pt"/>                                                                                    | distribution of <img alt="$\mathcal{Z}$" src="svgs/a51b274f1c4520f1cc0b3ee59d7a38e1.svg" align="middle" width="13.21921094999999pt" height="22.465723500000017pt"/>                         | `\mathcal{D}`      | `\fD`                 |
| <img alt="$S=\{\mathbf{z}_i\}_{i=1}^n$" src="svgs/ca9960bb6ef57ee579f56f77ecf902ea.svg" align="middle" width="84.55283099999998pt" height="24.65753399999998pt"/>                                                                     | <img alt="$=\{(\mathbf{x}_i,\mathbf{y}_i)\}_{i=1}^n$" src="svgs/f3cf7f1fdf7efb2d62b4e441a52da53d.svg" align="middle" width="106.0759128pt" height="24.65753399999998pt"/> sample set |
| <img alt="$L_S(\mathbf{\theta})$" src="svgs/800fde4ef8c4cf09b319be12c03a1d50.svg" align="middle" width="41.66906699999999pt" height="24.65753399999998pt"/>, <img alt="$L_{n}(\mathbf{\theta})$" src="svgs/db1a7f2c9f5962f1cd4cb0b222b5e99a.svg" align="middle" width="41.094140999999986pt" height="24.65753399999998pt"/>, <img alt="$R_n(\mathbf{\theta})$" src="svgs/0d8570bb69d1594ab193d13900d33570.svg" align="middle" width="42.38837294999998pt" height="24.65753399999998pt"/>, <img alt="$R_S(\mathbf{\theta})$" src="svgs/51ac91bc610ae6bd004d35c9c484fa51.svg" align="middle" width="42.96329894999999pt" height="24.65753399999998pt"/> | empirical risk or training loss                       |
| <img alt="$L_D(\mathbf{\theta})$" src="svgs/8ff92880d0d2256ab6d0bcd195021b66.svg" align="middle" width="44.07020639999999pt" height="24.65753399999998pt"/>                                                                           | population risk or expected loss                      |
| <img alt="$\sigma:\mathbb{R}\to\mathbb{R}^+$" src="svgs/7b80684f973f45e10fcfca816d6a9339.svg" align="middle" width="83.08763594999999pt" height="26.17730939999998pt"/>                                                               | activation function                                   | `\sigma`           |
| <img alt="$\mathbf{w}_j$" src="svgs/831047ac6f850b0d588c94d84fc6f4c1.svg" align="middle" width="19.75740524999999pt" height="14.611878600000017pt"/>                                                                                   | input weight                                          | `\bm{w}_j`         | `\mathbf{w}_j`        |
| <img alt="$a_j$" src="svgs/3fd897df5707a411645a54460183e3cd.svg" align="middle" width="14.793662399999992pt" height="14.15524440000002pt"/>                                                                                            | output weight                                         | `a_j`              |
| <img alt="$b_j$" src="svgs/2020a79c00e140ee1a054ecab57a289c.svg" align="middle" width="13.15930604999999pt" height="22.831056599999986pt"/>                                                                                            | bias term                                             | `b_j`              |
| <img alt="$f_{\mathbf{\theta}}(\mathbf{x})$" src="svgs/6cec5e8c8ed76bb809fd3d23b5afb245.svg" align="middle" width="38.24770454999999pt" height="24.65753399999998pt"/> or <img alt="$f(\mathbf{x};\mathbf{\theta})$" src="svgs/cf2dd6c8d8fd7e223ded07a2d09f5ff4.svg" align="middle" width="48.05937344999999pt" height="24.65753399999998pt"/>                             | neural network                                        | `f_{\bm{\theta}}`  | `f_{\mathbf{\theta}}` |
| <img alt="$\sum_{j=1}^{m} a_j \sigma (\mathbf{w}_j\cdot \mathbf{x} + b_j)$" src="svgs/16eb06f0dcb136d21db33bb43d98ea02.svg" align="middle" width="158.54628405pt" height="26.438629799999987pt"/>                                 | two-layer neural network                              |
| <img alt="$\text{VCdim}(\mathcal{H}$" src="svgs/cbd1997a81f60dcf0dee49de0a6b2916.svg" align="middle" width="71.57555459999999pt" height="24.65753399999998pt"/>)                                                                      | VC-dimension of <img alt="$\mathcal{H}$" src="svgs/8209c0f8b3c5233ea2e20dae55588c43.svg" align="middle" width="14.041179899999989pt" height="22.465723500000017pt"/>                         |
| <img alt="$\text{Rad}(\mathcal{H}\circ S)$" src="svgs/fb31f87b5e59405971bf46febe5cf39a.svg" align="middle" width="82.83105434999999pt" height="24.65753399999998pt"/>, <img alt="$\text{Rad}_{S}(\mathcal{H})$" src="svgs/a97f52c5cd9cca059cf894cda88dd8fa.svg" align="middle" width="65.80156769999999pt" height="24.65753399999998pt"/>                                  | Rademacher complexity of <img alt="$\mathcal{H}$" src="svgs/8209c0f8b3c5233ea2e20dae55588c43.svg" align="middle" width="14.041179899999989pt" height="22.465723500000017pt"/> on <img alt="$S$" src="svgs/e257acd1ccbe7fcb654708f1a866bfe9.svg" align="middle" width="11.027402099999989pt" height="22.465723500000017pt"/>         |
| <img alt="${\rm Rad}_{n} (\mathcal{H})$" src="svgs/63a818f7e0cbcc1f4187230c7a542414.svg" align="middle" width="65.22664169999999pt" height="24.65753399999998pt"/>                                                                    | Rademacher complexity over samples of size <img alt="$n$" src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg" align="middle" width="9.86687624999999pt" height="14.15524440000002pt"/>        |
| <img alt="$\text{GD}$" src="svgs/78bdfab9732d8b1d617479143855bbe5.svg" align="middle" width="25.45669169999999pt" height="22.465723500000017pt"/>                                                                                      | gradient descent                                      |
| <img alt="$\text{SGD}$" src="svgs/5075db15f12da3130c3137c0b7da7111.svg" align="middle" width="34.58913974999999pt" height="22.465723500000017pt"/>                                                                                     | stochastic gradient descent                           |
| <img alt="$B$" src="svgs/61e84f854bc6258d4108d08d4c4a0852.svg" align="middle" width="13.29340979999999pt" height="22.465723500000017pt"/>                                                                                              | a batch set                                           | `B`                |                       |
| <img alt="$\vert B\vert$" src="svgs/3c9e5bf9e3b2b5ea9e304df25541f33c.svg" align="middle" width="22.42585124999999pt" height="24.65753399999998pt"/>                                                                                   | batch size                                            | `b`                |                       |
| <img alt="$\eta$" src="svgs/1d0496971a2775f4887d1df25cea4f7e.svg" align="middle" width="8.751954749999989pt" height="14.15524440000002pt"/>                                                                                           | learning rate                                         | `\eta`             |
| <img alt="$\mathbf{k}$" src="svgs/9d152c065089da4147fb86e392670ac8.svg" align="middle" width="9.97711604999999pt" height="22.831056599999986pt"/>                                                                                     | discretized frequency                                 | `\bm{k}`           | `\mathbf{k}`          |
| <img alt="$\mathbf{\xi}$" src="svgs/8107e598d348f99cde7fc8b59875b9b0.svg" align="middle" width="7.94809454999999pt" height="22.831056599999986pt"/>                                                                                   | continuous frequency                                  | `\bm{\xi}`         | `\mathbf{xi}`         |  |
| <img alt="$*$" src="svgs/7c74eeb32158ff7c4f67d191b95450fb.svg" align="middle" width="8.219209349999991pt" height="15.296829900000011pt"/>                                                                                              | convolution operation                                 | `*`                |

## L-layer neural network

| symbol                                  | meaning                                                                                                                                | Latex          | simplied           |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | -------------- | ------------------ |
| <img alt="$d$" src="svgs/2103f85b8b1477f430fc407cad462224.svg" align="middle" width="8.55596444999999pt" height="22.831056599999986pt"/>                                     | input dimension                                                                                                                        | `d`            |                    |
| <img alt="$d_{\text{o}}$" src="svgs/8f865e6069b66d7a0e8a1f6f72400f41.svg" align="middle" width="15.10851044999999pt" height="22.831056599999986pt"/>                          | output dimension                                                                                                                       | `d_{\rm o}`    |                    |
| <img alt="$m_l$" src="svgs/882b3bd64477a9a54f36bf7891ca71d7.svg" align="middle" width="18.656888249999994pt" height="14.15524440000002pt"/>                                   | the number of <img alt="$l$" src="svgs/2f2322dff5bde89c37bcae4116fe20a8.svg" align="middle" width="5.2283516999999895pt" height="22.831056599999986pt"/>-th layer neuron, <img alt="$m_0=d$" src="svgs/dc5047d443cccd6a928b207b5d6ecf0b.svg" align="middle" width="52.281154199999996pt" height="22.831056599999986pt"/>, <img alt="$m_{L} = d_{\text{o}}$" src="svgs/8cbde3e6dedae95800eb714cf9e5e4b4.svg" align="middle" width="61.29947999999999pt" height="22.831056599999986pt"/>                                                                     | `m_l`          |
| <img alt="$\mathbf{W}^{[l]}$" src="svgs/4ebaf286ecf805798e1878fd26409f83.svg" align="middle" width="31.472604899999986pt" height="29.190975000000005pt"/>                      | the <img alt="$l$" src="svgs/2f2322dff5bde89c37bcae4116fe20a8.svg" align="middle" width="5.2283516999999895pt" height="22.831056599999986pt"/>-th layer weight                                                                                                                | `\bm{W}^{[l]}` | `\mathbf{W}^{[l]}` |
| <img alt="$\mathbf{b}^{[l]}$" src="svgs/56d50f1d4ef20c81ca02db009278a2ce.svg" align="middle" width="22.168985849999988pt" height="29.190975000000005pt"/>                      | the <img alt="$l$" src="svgs/2f2322dff5bde89c37bcae4116fe20a8.svg" align="middle" width="5.2283516999999895pt" height="22.831056599999986pt"/>-th layer bias term                                                                                                             | `\bm{b}^{[l]}` | `\mathbf{b}^{[l]}` |
| <img alt="$\circ$" src="svgs/c0463eeb4772bfde779c20d52901d01b.svg" align="middle" width="8.219209349999991pt" height="14.611911599999981pt"/>                                 | entry-wise operation                                                                                                                   | `\circ`        |
| <img alt="$\sigma:\mathbb{R}\to\mathbb{R}^+$" src="svgs/7b80684f973f45e10fcfca816d6a9339.svg" align="middle" width="83.08763594999999pt" height="26.17730939999998pt"/>      | activation function                                                                                                                    | `\sigma`       |
| <img alt="$\mathbf{\theta}$" src="svgs/6fccf0465699020081a15631f4a45ae1.svg" align="middle" width="8.17352744999999pt" height="22.831056599999986pt"/>                       | <img alt="$=(\mathbf{W}^{[0]},\ldots,\mathbf{W}^{[L-1]},\mathbf{b}^{[0]},\ldots,\mathbf{b}^{[L-1]})$" src="svgs/67a750d5b1df671b5a968a0dfe2f2cbd.svg" align="middle" width="268.97215124999997pt" height="29.190975000000005pt"/>, parameters                                 | `\bm{\theta}`  | `\mathbf{\theta}`  |
| <img alt="$f_{\mathbf{\theta}}^{[0]}(\mathbf{x})$" src="svgs/363347c50af93c2d3e4de844602aa1cf.svg" align="middle" width="47.39738354999999pt" height="34.337843099999986pt"/> | <img alt="$=\mathbf{x}$" src="svgs/a78ea489a342afd0c1cb2a77480f9cef.svg" align="middle" width="27.32864804999999pt" height="14.611878600000017pt"/>                                                                                                                          |
| <img alt="$f_{\mathbf{\theta}}^{[l]}(\mathbf{x})$" src="svgs/1388c6341f8133d3f2a6cab8a915880a.svg" align="middle" width="45.06860819999999pt" height="34.337843099999986pt"/> | <img alt="$=\sigma\circ(\mathbf{W}^{[l-1]} f_{\mathbf{\theta}}^{[l-1]}(\mathbf{x}) + \mathbf{b}^{[l-1]})$" src="svgs/e3f16cf9944326af1449c40e03c01159.svg" align="middle" width="226.56959984999997pt" height="34.337843099999986pt"/>, <img alt="$l$" src="svgs/2f2322dff5bde89c37bcae4116fe20a8.svg" align="middle" width="5.2283516999999895pt" height="22.831056599999986pt"/>-th layer output                   |
| <img alt="$f_{\mathbf{\theta}}(\mathbf{x})$" src="svgs/6cec5e8c8ed76bb809fd3d23b5afb245.svg" align="middle" width="38.24770454999999pt" height="24.65753399999998pt"/>       | <img alt="$=f_{\mathbf{\theta}}^{[L]}(\mathbf{x})=\mathbf{W}^{[L-1]} f_{\mathbf{\theta}}^{[L-1]}(\mathbf{x}) + \mathbf{b}^{[L-1]}$" src="svgs/fa0265e13fc6050820383d31af4148a2.svg" align="middle" width="273.61889444999997pt" height="34.337843099999986pt"/>, <img alt="$L$" src="svgs/ddcb483302ed36a59286424aa5e0be17.svg" align="middle" width="11.18724254999999pt" height="22.465723500000017pt"/>-layer NN |

# How to cite

Please cite this repository in your publications if it helps your research.
 ```
 @misc{beijing2020Suggested,
   title = {Suggested Notation for Machine Learning},
   author = {Beijing Academy of Artificial Intelligence},
   howpublished = {\url{https://github.com/Mayuyu/suggested-notation-for-machine-learning}},
   year=2020
}
 ```
# Acknowledgements

Chenglong Bao (Tsinghua), Zhengdao Chen (NYU), Bin Dong (Peking), Weinan E (Princeton), Quanquan Gu (UCLA), Kaizhu Huang (XJTLU), Shi Jin (SJTU), Jian Li (Tsinghua), Lei Li (SJTU), Tiejun Li (Peking), Zhenguo Li (Huawei), Zhemin Li (NUDT), Shaobo Lin (XJTU), Ziqi Liu (CSRC), Zichao Long (Peking), Chao Ma (Princeton), Chao Ma (SJTU), Yuheng Ma (WHU), Dengyu Meng (XJTU), Wang Miao (Peking), Pingbing Ming (CAS), Zuoqiang Shi (Tsinghua), Jihong Wang (CSRC), Liwei Wang (Peking), Bican Xia (Peking), Zhouwang Yang (USTC), Haijun Yu (CAS), Yang Yuan (Tsinghua), Cheng Zhang (Peking), Lulu Zhang (SJTU), Jiwei Zhang (WHU), Pingwen Zhang (Peking), Xiaoqun Zhang (SJTU), Chengchao Zhao (CSRC), Zhanxing Zhu (Peking), Chuan Zhou (CAS), Xiang Zhou (cityU).
