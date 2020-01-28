# Suggested Notation for Machine Learning

> This introduces a suggestion of mathematical notation protocol for machine learning.

The field of machine learning is evolving rapidly in recent years. Communication between different researchers and research groups becomes increasingly important. A key challenge for communication arises from inconsistent notation usages among different papers. This proposal suggests a standard for commonly used mathematical notation for machine learning. In this first version, only some notation are mentioned and more notation are left to be done. This proposal will be regularly updated based on the progress of the field. We look forward to more suggestions to improve this proposal in future versions.

## Tabel of Contents

- [A Proposal on Standard Notation for Machine Learning](#a-proposal-on-standard-notation-for-machine-learning)
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

Dataset <img src="/tex/a62b8a89e44f4588d548edf23e5964bd.svg?invert_in_darkmode&sanitize=true" align=middle width=196.01672969999998pt height=24.65753399999998pt/> is sampled from a distribution <img src="/tex/eaf85f2b753a4c7585def4cc7ecade43.svg?invert_in_darkmode&sanitize=true" align=middle width=13.13706569999999pt height=22.465723500000017pt/> over a domain <img src="/tex/7b416174c3d0e087d28a4cc81bae17fd.svg?invert_in_darkmode&sanitize=true" align=middle width=81.69842999999999pt height=22.465723500000017pt/>.

- <img src="/tex/7da75f4e61cdeabf944740206b511812.svg?invert_in_darkmode&sanitize=true" align=middle width=14.132466149999988pt height=22.465723500000017pt/> is the instances domain (a set)
- <img src="/tex/fce9019a5e1fa63e079199cd9b11c55e.svg?invert_in_darkmode&sanitize=true" align=middle width=12.337954199999992pt height=22.465723500000017pt/> is the label domain (a set)
- <img src="/tex/7b416174c3d0e087d28a4cc81bae17fd.svg?invert_in_darkmode&sanitize=true" align=middle width=81.69842999999999pt height=22.465723500000017pt/> is the example domain (a set)

Usually, <img src="/tex/7da75f4e61cdeabf944740206b511812.svg?invert_in_darkmode&sanitize=true" align=middle width=14.132466149999988pt height=22.465723500000017pt/> is a subset of <img src="/tex/435f1061aa6f25938c3c3515c083d06c.svg?invert_in_darkmode&sanitize=true" align=middle width=18.71525699999999pt height=27.91243950000002pt/> and <img src="/tex/fce9019a5e1fa63e079199cd9b11c55e.svg?invert_in_darkmode&sanitize=true" align=middle width=12.337954199999992pt height=22.465723500000017pt/> is a subset of <img src="/tex/8cf8e83175c24764c15718de79a77a04.svg?invert_in_darkmode&sanitize=true" align=middle width=24.308956649999992pt height=27.91243950000002pt/>, where <img src="/tex/2103f85b8b1477f430fc407cad462224.svg?invert_in_darkmode&sanitize=true" align=middle width=8.55596444999999pt height=22.831056599999986pt/> is the input dimension, <img src="/tex/d36df281d4cb7796ad889d761f50712c.svg?invert_in_darkmode&sanitize=true" align=middle width=15.10851044999999pt height=22.831056599999986pt/> is the ouput dimension.

<img src="/tex/62bf1fa8e6d3a545fb832bd073c542dd.svg?invert_in_darkmode&sanitize=true" align=middle width=56.510581049999985pt height=22.831056599999986pt/> is the number of samples. Wihout specification, <img src="/tex/e257acd1ccbe7fcb654708f1a866bfe9.svg?invert_in_darkmode&sanitize=true" align=middle width=11.027402099999989pt height=22.465723500000017pt/> and <img src="/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/> are for the training set.

## Function

Hypothesis space is denoted by <img src="/tex/8209c0f8b3c5233ea2e20dae55588c43.svg?invert_in_darkmode&sanitize=true" align=middle width=14.041179899999989pt height=22.465723500000017pt/>. Hypothesis function is denoted by <img src="/tex/21516e47cad2991185d50716b54b7ccd.svg?invert_in_darkmode&sanitize=true" align=middle width=72.38002035pt height=24.65753399999998pt/> or <img src="/tex/cf2dd6c8d8fd7e223ded07a2d09f5ff4.svg?invert_in_darkmode&sanitize=true" align=middle width=48.05937344999999pt height=24.65753399999998pt/> with <img src="/tex/cddc713171a7d82a6a94778d48fa9dad.svg?invert_in_darkmode&sanitize=true" align=middle width=81.22459289999999pt height=22.831056599999986pt/>.

<img src="/tex/6fccf0465699020081a15631f4a45ae1.svg?invert_in_darkmode&sanitize=true" align=middle width=8.17352744999999pt height=22.831056599999986pt/> denotes the set of parameters of <img src="/tex/43263b5b62e73bb17cf793dc765b7083.svg?invert_in_darkmode&sanitize=true" align=middle width=14.66328269999999pt height=22.831056599999986pt/>.

If there exists a target function, it is denoted by <img src="/tex/baa7cc2f36af87229745595791fda227.svg?invert_in_darkmode&sanitize=true" align=middle width=16.55260694999999pt height=22.831056599999986pt/> or <img src="/tex/78d6c299386fc79de759e7449c5fba27.svg?invert_in_darkmode&sanitize=true" align=middle width=83.11394024999998pt height=22.831056599999986pt/> satisfying <img src="/tex/4d877e4e6f3a8a26978f0124709d5f9a.svg?invert_in_darkmode&sanitize=true" align=middle width=82.97739119999999pt height=24.65753399999998pt/> for <img src="/tex/ae697d8a49bffcb50cc01bc8a09826f7.svg?invert_in_darkmode&sanitize=true" align=middle width=82.19635874999999pt height=21.68300969999999pt/>.

## Loss function

Loss function, denoted by <img src="/tex/91248c7c221d6ed56762761f1038d39f.svg?invert_in_darkmode&sanitize=true" align=middle width=198.44712029999997pt height=24.65753399999998pt/> measures the difference between a predicted label and a true label, e.g.,

- <img src="/tex/e8831293b846e3a3799cd6a02e4a0cd9.svg?invert_in_darkmode&sanitize=true" align=middle width=17.73978854999999pt height=26.76175259999998pt/> loss: <img src="/tex/267cbf28f1c2d7238eeb97e3f0c38b68.svg?invert_in_darkmode&sanitize=true" align=middle width=160.66181009999997pt height=26.76175259999998pt/>, where <img src="/tex/02e9c08c82c033896f32f3bf6b2ebb59.svg?invert_in_darkmode&sanitize=true" align=middle width=70.62752894999998pt height=24.65753399999998pt/>. <img src="/tex/d59371cab861973036670c707757eb37.svg?invert_in_darkmode&sanitize=true" align=middle width=50.82761969999999pt height=24.65753399999998pt/> can also be written as <img src="/tex/2c7e0a944f5282a0a8ed736c8c2d32ad.svg?invert_in_darkmode&sanitize=true" align=middle width=59.05823879999999pt height=24.65753399999998pt/> for convenience.

Empirical risk or training loss for a set <img src="/tex/4509212c364bba86adda3e6c197be90c.svg?invert_in_darkmode&sanitize=true" align=middle width=120.72371684999999pt height=24.65753399999998pt/> is denoted by <img src="/tex/800fde4ef8c4cf09b319be12c03a1d50.svg?invert_in_darkmode&sanitize=true" align=middle width=41.66906699999999pt height=24.65753399999998pt/> or <img src="/tex/5b927af53cdeff9f6707e49201987ae0.svg?invert_in_darkmode&sanitize=true" align=middle width=41.094140999999986pt height=24.65753399999998pt/> or <img src="/tex/531ee2725a779e305466daa52a074798.svg?invert_in_darkmode&sanitize=true" align=middle width=44.413391549999986pt height=24.65753399999998pt/> or <img src="/tex/dff852c1eb319723d0ad74cb89e9d642.svg?invert_in_darkmode&sanitize=true" align=middle width=43.83846554999999pt height=24.65753399999998pt/>,

<p align="center"><img src="/tex/b84cae2647caeb5068108f22f3a902c2.svg?invert_in_darkmode&sanitize=true" align=middle width=197.29974825pt height=44.89738935pt/></p>

Without ambiguity, <img src="/tex/ddcb483302ed36a59286424aa5e0be17.svg?invert_in_darkmode&sanitize=true" align=middle width=11.18724254999999pt height=22.465723500000017pt/> is also used for <img src="/tex/a9c88395fd83bab6dca8216dd1842e98.svg?invert_in_darkmode&sanitize=true" align=middle width=19.88819414999999pt height=22.465723500000017pt/>.

The population risk or expected loss is denoted by

<p align="center"><img src="/tex/c3b0bdd54fe608c69bd26ca6b16aebe2.svg?invert_in_darkmode&sanitize=true" align=middle width=174.2308986pt height=16.438356pt/></p>

where <img src="/tex/02e9c08c82c033896f32f3bf6b2ebb59.svg?invert_in_darkmode&sanitize=true" align=middle width=70.62752894999998pt height=24.65753399999998pt/> follows the distribution <img src="/tex/eaf85f2b753a4c7585def4cc7ecade43.svg?invert_in_darkmode&sanitize=true" align=middle width=13.13706569999999pt height=22.465723500000017pt/>.

## Activation function

Activation function is denoted by <img src="/tex/b9b27f3deff0db82f962a8505706e620.svg?invert_in_darkmode&sanitize=true" align=middle width=32.16330314999999pt height=24.65753399999998pt/>.

**Example 1**. Some commonly used activation functions are

- <img src="/tex/d215872452b1ebb225b1e576edcb7879.svg?invert_in_darkmode&sanitize=true" align=middle width=208.48739834999995pt height=24.65753399999998pt/>
- <img src="/tex/a4d649352a659f0b7e0cc3b5b5670bd3.svg?invert_in_darkmode&sanitize=true" align=middle width=209.55236114999997pt height=43.42856099999997pt/>
- <img src="/tex/f64afc209f5ec2703004ba8141966a8f.svg?invert_in_darkmode&sanitize=true" align=middle width=109.13816759999997pt height=24.65753399999998pt/>
- <img src="/tex/0b557fdfdec1313c966fb5c9cfbee5a1.svg?invert_in_darkmode&sanitize=true" align=middle width=127.84794989999999pt height=24.65753399999998pt/>

## Two-layer neural network

The neuron number of the hidden layer is denoted by <img src="/tex/0e51a2dede42189d77627c4d742822c3.svg?invert_in_darkmode&sanitize=true" align=middle width=14.433101099999991pt height=14.15524440000002pt/>, The two-layer neural network is

<p align="center"><img src="/tex/bcbf19b6c41d3d362d805920319da7e6.svg?invert_in_darkmode&sanitize=true" align=middle width=206.10021794999997pt height=47.1348339pt/></p>

where <img src="/tex/8cda31ed38c6d59d14ebefa440099572.svg?invert_in_darkmode&sanitize=true" align=middle width=9.98290094999999pt height=14.15524440000002pt/> is the activation function, <img src="/tex/831047ac6f850b0d588c94d84fc6f4c1.svg?invert_in_darkmode&sanitize=true" align=middle width=19.75740524999999pt height=14.611878600000017pt/> is the input weight, <img src="/tex/3fd897df5707a411645a54460183e3cd.svg?invert_in_darkmode&sanitize=true" align=middle width=14.793662399999992pt height=14.15524440000002pt/> is the output weight, <img src="/tex/2020a79c00e140ee1a054ecab57a289c.svg?invert_in_darkmode&sanitize=true" align=middle width=13.15930604999999pt height=22.831056599999986pt/> is the bias term. We denote the set of parameters by

<p align="center"><img src="/tex/aa9b11397a4c6643cb07e6b991c9fc7e.svg?invert_in_darkmode&sanitize=true" align=middle width=292.7598531pt height=16.438356pt/></p>

## General deep neural network

The counting of the layer number excludes the input layer. A <img src="/tex/ddcb483302ed36a59286424aa5e0be17.svg?invert_in_darkmode&sanitize=true" align=middle width=11.18724254999999pt height=22.465723500000017pt/>-layer neural network is denoted by
<p align="center"><img src="/tex/26e99c0533a05446e1e6ce80dfaecfaa.svg?invert_in_darkmode&sanitize=true" align=middle width=652.6473376499999pt height=19.526994300000002pt/></p>

where <img src="/tex/865e70ab5839636feaab8a6745125c4a.svg?invert_in_darkmode&sanitize=true" align=middle width=120.62059019999998pt height=29.190975000000005pt/>, <img src="/tex/998139a600e0e2203867005393bb05b4.svg?invert_in_darkmode&sanitize=true" align=middle width=86.43477479999999pt height=29.190975000000005pt/>, <img src="/tex/b5142f01744a994ace1bc28b20b87eed.svg?invert_in_darkmode&sanitize=true" align=middle width=94.55845409999999pt height=22.831056599999986pt/>, <img src="/tex/366a5d477d66fb8c7c0d12af876c22e3.svg?invert_in_darkmode&sanitize=true" align=middle width=61.29947999999999pt height=22.831056599999986pt/>, <img src="/tex/8cda31ed38c6d59d14ebefa440099572.svg?invert_in_darkmode&sanitize=true" align=middle width=9.98290094999999pt height=14.15524440000002pt/> is a scalar function and "<img src="/tex/c0463eeb4772bfde779c20d52901d01b.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=14.611911599999981pt/>" means entry-wise operation. We denote the set of parameters by

<p align="center"><img src="/tex/4666610504d6d382d12463f8941f507c.svg?invert_in_darkmode&sanitize=true" align=middle width=360.8326854pt height=19.526994300000002pt/></p>

This can also be defined recursively,

<p align="center"><img src="/tex/60638ec6e9895b57f49ac860aa8d7c0f.svg?invert_in_darkmode&sanitize=true" align=middle width=83.85835424999999pt height=22.127694599999998pt/></p>
<p align="center"><img src="/tex/c6ed8c3c10c3702b34c8350acc55b0b8.svg?invert_in_darkmode&sanitize=true" align=middle width=401.2952526pt height=22.127694599999998pt/></p>
<p align="center"><img src="/tex/668c8dfd4833fded1b6f21375b5454d0.svg?invert_in_darkmode&sanitize=true" align=middle width=442.34552009999993pt height=22.127694599999998pt/></p>

## Complexity

The VC-dimension of a hypothesis class <img src="/tex/8209c0f8b3c5233ea2e20dae55588c43.svg?invert_in_darkmode&sanitize=true" align=middle width=14.041179899999989pt height=22.465723500000017pt/> is denoted as VCdim(<img src="/tex/8209c0f8b3c5233ea2e20dae55588c43.svg?invert_in_darkmode&sanitize=true" align=middle width=14.041179899999989pt height=22.465723500000017pt/>).

The Rademacher complexity of a hypothesis space <img src="/tex/8209c0f8b3c5233ea2e20dae55588c43.svg?invert_in_darkmode&sanitize=true" align=middle width=14.041179899999989pt height=22.465723500000017pt/> on a sample set <img src="/tex/e257acd1ccbe7fcb654708f1a866bfe9.svg?invert_in_darkmode&sanitize=true" align=middle width=11.027402099999989pt height=22.465723500000017pt/> is denoted by <img src="/tex/eb048e4d3123034dac2256effd67ad18.svg?invert_in_darkmode&sanitize=true" align=middle width=65.98741049999998pt height=24.65753399999998pt/> or <img src="/tex/aba734a98a13ed03fa63549be906337b.svg?invert_in_darkmode&sanitize=true" align=middle width=65.80156769999999pt height=24.65753399999998pt/>. The complexity <img src="/tex/aba734a98a13ed03fa63549be906337b.svg?invert_in_darkmode&sanitize=true" align=middle width=65.80156769999999pt height=24.65753399999998pt/> is random because of the randomness of <img src="/tex/e257acd1ccbe7fcb654708f1a866bfe9.svg?invert_in_darkmode&sanitize=true" align=middle width=11.027402099999989pt height=22.465723500000017pt/>. The expectation of the empirical Rademacher complexity over all samples of size <img src="/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/> is denoted by

<p align="center"><img src="/tex/6a8c078cff5f892c588c9f5baeb0c908.svg?invert_in_darkmode&sanitize=true" align=middle width=177.9938688pt height=16.438356pt/></p>

## Training

The Gradient Descent is oftern denoted by GD. THe Stochastic Gradient Descent is often denoted by SGD.

A batch set is denoted by <img src="/tex/61e84f854bc6258d4108d08d4c4a0852.svg?invert_in_darkmode&sanitize=true" align=middle width=13.29340979999999pt height=22.465723500000017pt/> and the batch size is denoted by <img src="/tex/007b57eceda75cfb83dcf22bd67fada1.svg?invert_in_darkmode&sanitize=true" align=middle width=22.42585124999999pt height=24.65753399999998pt/>.

The learning rate is denoted by <img src="/tex/1d0496971a2775f4887d1df25cea4f7e.svg?invert_in_darkmode&sanitize=true" align=middle width=8.751954749999989pt height=14.15524440000002pt/>.

## Fourier Frequency

The discretized frequency is denoted by <img src="/tex/9d152c065089da4147fb86e392670ac8.svg?invert_in_darkmode&sanitize=true" align=middle width=9.97711604999999pt height=22.831056599999986pt/>, and the continuous frequency is denoted by <img src="/tex/8107e598d348f99cde7fc8b59875b9b0.svg?invert_in_darkmode&sanitize=true" align=middle width=7.94809454999999pt height=22.831056599999986pt/>.

## Convolution

The convolution operation is denoted by <img src="/tex/7c74eeb32158ff7c4f67d191b95450fb.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=15.296829900000011pt/>.

## Notation table

| symbol | meaning | Latex | simplied |
| --------- | --------- | --------- | --------- |
| <img src="/tex/b0ea07dc5c00127344a1cad40467b8de.svg?invert_in_darkmode&sanitize=true" align=middle width=9.97711604999999pt height=14.611878600000017pt/> | input | `\bm{x}` | `\mathbf{x}` |
| <img src="/tex/1da18d2de6d16a18e780cd6c435a2936.svg?invert_in_darkmode&sanitize=true" align=middle width=10.239687149999991pt height=14.611878600000017pt/> | output, label | `\bm{y}` | `\vy` |
| <img src="/tex/2103f85b8b1477f430fc407cad462224.svg?invert_in_darkmode&sanitize=true" align=middle width=8.55596444999999pt height=22.831056599999986pt/> | input dimension | `d` |  |
| <img src="/tex/8f865e6069b66d7a0e8a1f6f72400f41.svg?invert_in_darkmode&sanitize=true" align=middle width=15.10851044999999pt height=22.831056599999986pt/> | output dimension | `d_{\rm o}` |  |
| <img src="/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/> | number of samples | `n`  |
| <img src="/tex/7da75f4e61cdeabf944740206b511812.svg?invert_in_darkmode&sanitize=true" align=middle width=14.132466149999988pt height=22.465723500000017pt/> | instances domain (a set)|`\mathcal{X}` | `\fX` |
| <img src="/tex/fce9019a5e1fa63e079199cd9b11c55e.svg?invert_in_darkmode&sanitize=true" align=middle width=12.337954199999992pt height=22.465723500000017pt/> | labels domain (a set)| `\mathcal{Y}` | `\fY` |
| <img src="/tex/a51b274f1c4520f1cc0b3ee59d7a38e1.svg?invert_in_darkmode&sanitize=true" align=middle width=13.21921094999999pt height=22.465723500000017pt/> | <img src="/tex/01c5b24158b66a590fee8a779d98c7b6.svg?invert_in_darkmode&sanitize=true" align=middle width=63.913139399999984pt height=22.465723500000017pt/> example domain|`\mathcal{Z}` | `\fZ` |
| <img src="/tex/8209c0f8b3c5233ea2e20dae55588c43.svg?invert_in_darkmode&sanitize=true" align=middle width=14.041179899999989pt height=22.465723500000017pt/>| hypothesis space (a set)| `\mathcal{H}` | `\mathcal{H}` |
| <img src="/tex/6fccf0465699020081a15631f4a45ae1.svg?invert_in_darkmode&sanitize=true" align=middle width=8.17352744999999pt height=22.831056599999986pt/> | a set of parameters | `\bm{\theta}` | `\mathbf{\theta}` |
| <img src="/tex/09eb4790c37751288e62ebbc4a99190d.svg?invert_in_darkmode&sanitize=true" align=middle width=81.22459289999999pt height=22.831056599999986pt/> | hypothesis function | `\f_{\bm{\theta}}` | `f_{\mathbf{\theta}}` |
| <img src="/tex/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode&sanitize=true" align=middle width=9.81741584999999pt height=22.831056599999986pt/> or <img src="/tex/a48536501c8099f27b17f0147fbb43a0.svg?invert_in_darkmode&sanitize=true" align=middle width=83.11394024999998pt height=22.831056599999986pt/> | target function  | `f,f^*` |
| <img src="/tex/64e4c6b981b5ba045a299994ba55aabb.svg?invert_in_darkmode&sanitize=true" align=middle width=115.43348354999998pt height=26.17730939999998pt/> | loss function | `\ell` |
| <img src="/tex/eaf85f2b753a4c7585def4cc7ecade43.svg?invert_in_darkmode&sanitize=true" align=middle width=13.13706569999999pt height=22.465723500000017pt/> | distribution of <img src="/tex/a51b274f1c4520f1cc0b3ee59d7a38e1.svg?invert_in_darkmode&sanitize=true" align=middle width=13.21921094999999pt height=22.465723500000017pt/> | `\mathcal{D}` | `\fD` |
| <img src="/tex/ca9960bb6ef57ee579f56f77ecf902ea.svg?invert_in_darkmode&sanitize=true" align=middle width=84.55283099999998pt height=24.65753399999998pt/> | <img src="/tex/f3cf7f1fdf7efb2d62b4e441a52da53d.svg?invert_in_darkmode&sanitize=true" align=middle width=106.0759128pt height=24.65753399999998pt/> sample set |
<img src="/tex/800fde4ef8c4cf09b319be12c03a1d50.svg?invert_in_darkmode&sanitize=true" align=middle width=41.66906699999999pt height=24.65753399999998pt/>, <img src="/tex/db1a7f2c9f5962f1cd4cb0b222b5e99a.svg?invert_in_darkmode&sanitize=true" align=middle width=41.094140999999986pt height=24.65753399999998pt/>, <img src="/tex/673efcd2398003146fd7fcb907a9a5ae.svg?invert_in_darkmode&sanitize=true" align=middle width=43.83846554999999pt height=24.65753399999998pt/>, <img src="/tex/371b55d5394da6dfa43a7b402840faeb.svg?invert_in_darkmode&sanitize=true" align=middle width=44.413391549999986pt height=24.65753399999998pt/> | empirical risk or training loss |
| <img src="/tex/8ff92880d0d2256ab6d0bcd195021b66.svg?invert_in_darkmode&sanitize=true" align=middle width=44.07020639999999pt height=24.65753399999998pt/> | population risk or expected loss |
| <img src="/tex/7b80684f973f45e10fcfca816d6a9339.svg?invert_in_darkmode&sanitize=true" align=middle width=83.08763594999999pt height=26.17730939999998pt/>| activation function | `\sigma` |
| <img src="/tex/831047ac6f850b0d588c94d84fc6f4c1.svg?invert_in_darkmode&sanitize=true" align=middle width=19.75740524999999pt height=14.611878600000017pt/> | input weight | `\bm{w}_j` | `\mathbf{w}_j` |
| <img src="/tex/3fd897df5707a411645a54460183e3cd.svg?invert_in_darkmode&sanitize=true" align=middle width=14.793662399999992pt height=14.15524440000002pt/> | output weight | `a_j` |
| <img src="/tex/2020a79c00e140ee1a054ecab57a289c.svg?invert_in_darkmode&sanitize=true" align=middle width=13.15930604999999pt height=22.831056599999986pt/> | bias term | `b_j` |
| <img src="/tex/6cec5e8c8ed76bb809fd3d23b5afb245.svg?invert_in_darkmode&sanitize=true" align=middle width=38.24770454999999pt height=24.65753399999998pt/> or <img src="/tex/cf2dd6c8d8fd7e223ded07a2d09f5ff4.svg?invert_in_darkmode&sanitize=true" align=middle width=48.05937344999999pt height=24.65753399999998pt/> | neural network |`f_{\bm{\theta}}`|`f_{\mathbf{\theta}}`|
| <img src="/tex/16eb06f0dcb136d21db33bb43d98ea02.svg?invert_in_darkmode&sanitize=true" align=middle width=158.54628405pt height=26.438629799999987pt/>| two-layer neural network |
| <img src="/tex/cbd1997a81f60dcf0dee49de0a6b2916.svg?invert_in_darkmode&sanitize=true" align=middle width=71.57555459999999pt height=24.65753399999998pt/>) | VC-dimension of <img src="/tex/8209c0f8b3c5233ea2e20dae55588c43.svg?invert_in_darkmode&sanitize=true" align=middle width=14.041179899999989pt height=22.465723500000017pt/> |
| <img src="/tex/fb31f87b5e59405971bf46febe5cf39a.svg?invert_in_darkmode&sanitize=true" align=middle width=82.83105434999999pt height=24.65753399999998pt/>, <img src="/tex/a97f52c5cd9cca059cf894cda88dd8fa.svg?invert_in_darkmode&sanitize=true" align=middle width=65.80156769999999pt height=24.65753399999998pt/>| Rademacher complexity of <img src="/tex/8209c0f8b3c5233ea2e20dae55588c43.svg?invert_in_darkmode&sanitize=true" align=middle width=14.041179899999989pt height=22.465723500000017pt/> on <img src="/tex/e257acd1ccbe7fcb654708f1a866bfe9.svg?invert_in_darkmode&sanitize=true" align=middle width=11.027402099999989pt height=22.465723500000017pt/>|
| <img src="/tex/63a818f7e0cbcc1f4187230c7a542414.svg?invert_in_darkmode&sanitize=true" align=middle width=65.22664169999999pt height=24.65753399999998pt/>| Rademacher complexity over samples of size <img src="/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/> |
| <img src="/tex/78bdfab9732d8b1d617479143855bbe5.svg?invert_in_darkmode&sanitize=true" align=middle width=25.45669169999999pt height=22.465723500000017pt/> | gradient descent |
| <img src="/tex/5075db15f12da3130c3137c0b7da7111.svg?invert_in_darkmode&sanitize=true" align=middle width=34.58913974999999pt height=22.465723500000017pt/> | stochastic gradient descent |
| <img src="/tex/61e84f854bc6258d4108d08d4c4a0852.svg?invert_in_darkmode&sanitize=true" align=middle width=13.29340979999999pt height=22.465723500000017pt/> | a batch set | `B` |
| <img src="/tex/594312d6ed7a82116746ac4f826b6095.svg?invert_in_darkmode&sanitize=true" align=middle width=4.5662248499999905pt height=24.65753399999998pt/> | batch size | `b` |
| <img src="/tex/1d0496971a2775f4887d1df25cea4f7e.svg?invert_in_darkmode&sanitize=true" align=middle width=8.751954749999989pt height=14.15524440000002pt/> | learning rate |`\eta`|
| <img src="/tex/9d152c065089da4147fb86e392670ac8.svg?invert_in_darkmode&sanitize=true" align=middle width=9.97711604999999pt height=22.831056599999986pt/> | discretized frequency | `\bm{k}` | `\mathbf{k}` |
| <img src="/tex/8107e598d348f99cde7fc8b59875b9b0.svg?invert_in_darkmode&sanitize=true" align=middle width=7.94809454999999pt height=22.831056599999986pt/> | continuous frequency | `\bm{\xi}` | `\mathbf{x}i` | |
| <img src="/tex/7c74eeb32158ff7c4f67d191b95450fb.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=15.296829900000011pt/> | convolution operation | `*` |

## L-layer neural network

| symbol | meaning | Latex | simplied |
| --------- | --------- | --------- | --------- |
| <img src="/tex/2103f85b8b1477f430fc407cad462224.svg?invert_in_darkmode&sanitize=true" align=middle width=8.55596444999999pt height=22.831056599999986pt/> | input dimension | `d` |  |
| <img src="/tex/8f865e6069b66d7a0e8a1f6f72400f41.svg?invert_in_darkmode&sanitize=true" align=middle width=15.10851044999999pt height=22.831056599999986pt/> | output dimension |`d_{\rm o}` |  |
| <img src="/tex/882b3bd64477a9a54f36bf7891ca71d7.svg?invert_in_darkmode&sanitize=true" align=middle width=18.656888249999994pt height=14.15524440000002pt/> | the number of <img src="/tex/2f2322dff5bde89c37bcae4116fe20a8.svg?invert_in_darkmode&sanitize=true" align=middle width=5.2283516999999895pt height=22.831056599999986pt/>-th layer neuron, <img src="/tex/dc5047d443cccd6a928b207b5d6ecf0b.svg?invert_in_darkmode&sanitize=true" align=middle width=52.281154199999996pt height=22.831056599999986pt/>, <img src="/tex/8cbde3e6dedae95800eb714cf9e5e4b4.svg?invert_in_darkmode&sanitize=true" align=middle width=61.29947999999999pt height=22.831056599999986pt/> | `m_l` |
| <img src="/tex/4ebaf286ecf805798e1878fd26409f83.svg?invert_in_darkmode&sanitize=true" align=middle width=31.472604899999986pt height=29.190975000000005pt/> | the <img src="/tex/2f2322dff5bde89c37bcae4116fe20a8.svg?invert_in_darkmode&sanitize=true" align=middle width=5.2283516999999895pt height=22.831056599999986pt/>-th layer weight | `\bm{W}^{[l]}` | `\mathbf{W}^{[l]}` |
| <img src="/tex/56d50f1d4ef20c81ca02db009278a2ce.svg?invert_in_darkmode&sanitize=true" align=middle width=22.168985849999988pt height=29.190975000000005pt/> | the <img src="/tex/2f2322dff5bde89c37bcae4116fe20a8.svg?invert_in_darkmode&sanitize=true" align=middle width=5.2283516999999895pt height=22.831056599999986pt/>-th layer bias term |`\bm{b}^{[l]}`| `\mathbf{b}^{[l]}` |
| <img src="/tex/c0463eeb4772bfde779c20d52901d01b.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=14.611911599999981pt/> | entry-wise operation | `\circ` |
| <img src="/tex/7b80684f973f45e10fcfca816d6a9339.svg?invert_in_darkmode&sanitize=true" align=middle width=83.08763594999999pt height=26.17730939999998pt/> | activation function |`\sigma`|
| <img src="/tex/6fccf0465699020081a15631f4a45ae1.svg?invert_in_darkmode&sanitize=true" align=middle width=8.17352744999999pt height=22.831056599999986pt/> | <img src="/tex/67a750d5b1df671b5a968a0dfe2f2cbd.svg?invert_in_darkmode&sanitize=true" align=middle width=268.97215124999997pt height=29.190975000000005pt/>,  parameters|`\bm{\theta}`|`\mathbf{\theta}`|
| <img src="/tex/363347c50af93c2d3e4de844602aa1cf.svg?invert_in_darkmode&sanitize=true" align=middle width=47.39738354999999pt height=34.337843099999986pt/>|<img src="/tex/a78ea489a342afd0c1cb2a77480f9cef.svg?invert_in_darkmode&sanitize=true" align=middle width=27.32864804999999pt height=14.611878600000017pt/>|
| <img src="/tex/1388c6341f8133d3f2a6cab8a915880a.svg?invert_in_darkmode&sanitize=true" align=middle width=45.06860819999999pt height=34.337843099999986pt/>|<img src="/tex/e3f16cf9944326af1449c40e03c01159.svg?invert_in_darkmode&sanitize=true" align=middle width=226.56959984999997pt height=34.337843099999986pt/>,  <img src="/tex/2f2322dff5bde89c37bcae4116fe20a8.svg?invert_in_darkmode&sanitize=true" align=middle width=5.2283516999999895pt height=22.831056599999986pt/>-th  layer output |
| <img src="/tex/6cec5e8c8ed76bb809fd3d23b5afb245.svg?invert_in_darkmode&sanitize=true" align=middle width=38.24770454999999pt height=24.65753399999998pt/>|<img src="/tex/fa0265e13fc6050820383d31af4148a2.svg?invert_in_darkmode&sanitize=true" align=middle width=273.61889444999997pt height=34.337843099999986pt/>,  <img src="/tex/ddcb483302ed36a59286424aa5e0be17.svg?invert_in_darkmode&sanitize=true" align=middle width=11.18724254999999pt height=22.465723500000017pt/>-layer NN|

# Acknowledgements

Chenglong Bao (Tsinghua), Zhengdao Chen (NYU), Bin Dong (Peking), Weinan E (Princeton),  Quanquan Gu (UCLA), Kaizhu Huang (XJTLU), Shi Jin (SJTU), Jian Li (Tsinghua), Lei Li (SJTU), Tiejun Li (Peking),   Zhenguo Li (Huawei), Zhemin Li (NUDT), Shaobo Lin (XJTU), Ziqi Liu (CSRC),  Zichao Long (Peking), Chao Ma (Princeton),  Chao Ma (SJTU), Yuheng Ma (WHU),    Dengyu Meng (XJTU), Wang Miao (Peking),  Pingbing Ming (CAS), Zuoqiang Shi (Tsinghua), Jihong Wang (CSRC), Liwei Wang (Peking), Bican Xia (Peking), Zhouwang Yang (USTC),  Haijun Yu (CAS),  Yang Yuan  (Tsinghua),  Cheng Zhang (Peking),  Lulu Zhang (SJTU), Jiwei Zhang  (WHU),   Pingwen Zhang (Peking), Xiaoqun Zhang (SJTU),  Chengchao Zhao (CSRC), Zhanxing Zhu (Peking), Chuan Zhou (CAS),  Xiang Zhou (cityU). 
