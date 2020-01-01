# A Proposal on Standard Mathematical Notations of Machine Learning

> This introduces a suggestion of mathematical notation protocol for machine learning.

Machine learning becomes increasingly important in various applications and scientific problems. A major challenge in commuicating the progress of machine learning from different perspectives arises from the notation usage. This note introduces a suggestion of mathematical notations for machine learning. We will regularly update this note. In the current version, we only introduce some very basec and commonly used notations. We look forward to more suggestions for refining this note in the future versions.

## Tabel of Contents

- [A Proposal on Standard Mathematical Notations of Machine Learning](#a-proposal-on-standard-mathematical-notations-of-machine-learning)
  - [Tabel of Contents](#tabel-of-contents)
  - [Dataset](#dataset)
  - [Function](#function)
  - [Loss function](#loss-function)
  - [Activation function](#activation-function)
  - [Two-layer neural network](#two-layer-neural-network)
  - [General deep neural network](#general-deep-neural-network)
  - [Complexity](#complexity)
  - [Training](#training)
  - [Gram matrix](#gram-matrix)

## Dataset

Dataset <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/a62b8a89e44f4588d548edf23e5964bd.svg?invert_in_darkmode" align=middle width=196.01672969999998pt height=24.65753399999998pt/> is sampled from a distribution <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/eaf85f2b753a4c7585def4cc7ecade43.svg?invert_in_darkmode" align=middle width=13.13706569999999pt height=22.465723500000017pt/> over a domain <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/7b416174c3d0e087d28a4cc81bae17fd.svg?invert_in_darkmode" align=middle width=81.69842999999999pt height=22.465723500000017pt/>.

- <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/7da75f4e61cdeabf944740206b511812.svg?invert_in_darkmode" align=middle width=14.132466149999988pt height=22.465723500000017pt/> is the instances domain (a set)
- <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/fce9019a5e1fa63e079199cd9b11c55e.svg?invert_in_darkmode" align=middle width=12.337954199999992pt height=22.465723500000017pt/> is the label domain (a set)
- <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/7b416174c3d0e087d28a4cc81bae17fd.svg?invert_in_darkmode" align=middle width=81.69842999999999pt height=22.465723500000017pt/> is the examples domain (a set)

Usually, <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/7da75f4e61cdeabf944740206b511812.svg?invert_in_darkmode" align=middle width=14.132466149999988pt height=22.465723500000017pt/> is a subset of <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/435f1061aa6f25938c3c3515c083d06c.svg?invert_in_darkmode" align=middle width=18.71525699999999pt height=27.91243950000002pt/> and <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/fce9019a5e1fa63e079199cd9b11c55e.svg?invert_in_darkmode" align=middle width=12.337954199999992pt height=22.465723500000017pt/> is a subset of <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/02e80e152e955a116803cc7641b9162f.svg?invert_in_darkmode" align=middle width=24.308956649999992pt height=27.91243950000002pt/>, where <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/2103f85b8b1477f430fc407cad462224.svg?invert_in_darkmode" align=middle width=8.55596444999999pt height=22.831056599999986pt/> is the input dimension, <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/79d60be3e08ccb36240b095b32ae9a64.svg?invert_in_darkmode" align=middle width=15.10851044999999pt height=22.831056599999986pt/> is the ouput dimension.

<img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/54744f3dc5da6bfcfcefd9d907d8c772.svg?invert_in_darkmode" align=middle width=51.944338049999985pt height=24.65753399999998pt/> is the number of samples. Wihout other specified, <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/6f9bad7347b91ceebebd3ad7e6f6f2d1.svg?invert_in_darkmode" align=middle width=7.7054801999999905pt height=14.15524440000002pt/> and <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode" align=middle width=9.86687624999999pt height=14.15524440000002pt/> are for the training set.

## Function

Hypothesis space is denoted by <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/8209c0f8b3c5233ea2e20dae55588c43.svg?invert_in_darkmode" align=middle width=14.041179899999989pt height=22.465723500000017pt/>. Hypothesis function is denoted by <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/75c7b77601c2ddef75ad3aefe812bc54.svg?invert_in_darkmode" align=middle width=49.61747174999999pt height=22.831056599999986pt/> with <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/cddc713171a7d82a6a94778d48fa9dad.svg?invert_in_darkmode" align=middle width=81.22459289999999pt height=22.831056599999986pt/>.

<img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/6fccf0465699020081a15631f4a45ae1.svg?invert_in_darkmode" align=middle width=8.17352744999999pt height=22.831056599999986pt/> denotes the set of parameters of <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/43263b5b62e73bb17cf793dc765b7083.svg?invert_in_darkmode" align=middle width=14.66328269999999pt height=22.831056599999986pt/>.

The target function is denoted by <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/78d6c299386fc79de759e7449c5fba27.svg?invert_in_darkmode" align=middle width=83.11394024999998pt height=22.831056599999986pt/> satisfying <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/4d877e4e6f3a8a26978f0124709d5f9a.svg?invert_in_darkmode" align=middle width=82.97739119999999pt height=24.65753399999998pt/> for <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/ae697d8a49bffcb50cc01bc8a09826f7.svg?invert_in_darkmode" align=middle width=82.19635874999999pt height=21.68300969999999pt/>.

## Loss function

Loss function, denoted by <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/91248c7c221d6ed56762761f1038d39f.svg?invert_in_darkmode" align=middle width=198.44712029999997pt height=24.65753399999998pt/> which measures the difference between a predicted label and a true label, e.g.,

- <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/e8831293b846e3a3799cd6a02e4a0cd9.svg?invert_in_darkmode" align=middle width=17.73978854999999pt height=26.76175259999998pt/> loss: <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/267cbf28f1c2d7238eeb97e3f0c38b68.svg?invert_in_darkmode" align=middle width=160.66181009999997pt height=26.76175259999998pt/>, where <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/02e9c08c82c033896f32f3bf6b2ebb59.svg?invert_in_darkmode" align=middle width=70.62752894999998pt height=24.65753399999998pt/>. <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/d59371cab861973036670c707757eb37.svg?invert_in_darkmode" align=middle width=50.82761969999999pt height=24.65753399999998pt/> can also be written as <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/2c7e0a944f5282a0a8ed736c8c2d32ad.svg?invert_in_darkmode" align=middle width=59.05823879999999pt height=24.65753399999998pt/> for convenience.

Empirical risk or training loss is denoted by <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/800fde4ef8c4cf09b319be12c03a1d50.svg?invert_in_darkmode" align=middle width=41.66906699999999pt height=24.65753399999998pt/> or <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/74128b9582f1d2c317fd2246a8627c87.svg?invert_in_darkmode" align=middle width=41.094140999999986pt height=31.141535699999984pt/>,

<p align="center"><img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/4a6662a23b1b4c2c6c1b63c5acd2f072.svg?invert_in_darkmode" align=middle width=197.29974825pt height=44.89738935pt/></p>
without further explanation <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/ddcb483302ed36a59286424aa5e0be17.svg?invert_in_darkmode" align=middle width=11.18724254999999pt height=22.465723500000017pt/> will be used for <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/a9c88395fd83bab6dca8216dd1842e98.svg?invert_in_darkmode" align=middle width=19.88819414999999pt height=22.465723500000017pt/>.

The population risk or expected loss is denoted by

<p align="center"><img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/5b3179302b5b8db9e57b75cc82f1dd87.svg?invert_in_darkmode" align=middle width=174.2308986pt height=16.438356pt/></p>

## Activation function

Activation function is denoted by <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/b9b27f3deff0db82f962a8505706e620.svg?invert_in_darkmode" align=middle width=32.16330314999999pt height=24.65753399999998pt/>. Some commonly used activation functions are

- <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/64f24e3b330dcb54fca139691cc1e15e.svg?invert_in_darkmode" align=middle width=205.40529074999998pt height=24.65753399999998pt/>
- <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/f2278369cb2c60303e141dad34d9792a.svg?invert_in_darkmode" align=middle width=206.79562695pt height=43.42856099999997pt/>
- <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/3bfb1e52734c7cf05703e77d134d68e0.svg?invert_in_darkmode" align=middle width=107.08342919999998pt height=24.65753399999998pt/>
- <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/570159f5ed6441a3c08369b8efef46dc.svg?invert_in_darkmode" align=middle width=124.76584724999998pt height=24.65753399999998pt/>

## Two-layer neural network

THe neuron number of the hidden layer is denoted by <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/0e51a2dede42189d77627c4d742822c3.svg?invert_in_darkmode" align=middle width=14.433101099999991pt height=14.15524440000002pt/>,

<p align="center"><img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/bcbf19b6c41d3d362d805920319da7e6.svg?invert_in_darkmode" align=middle width=206.10021794999997pt height=47.1348339pt/></p>
where <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/8cda31ed38c6d59d14ebefa440099572.svg?invert_in_darkmode" align=middle width=9.98290094999999pt height=14.15524440000002pt/> is the activation function, <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/831047ac6f850b0d588c94d84fc6f4c1.svg?invert_in_darkmode" align=middle width=19.75740524999999pt height=14.611878600000017pt/> is the input weight, <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/3fd897df5707a411645a54460183e3cd.svg?invert_in_darkmode" align=middle width=14.793662399999992pt height=14.15524440000002pt/> is the output weight, <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/2020a79c00e140ee1a054ecab57a289c.svg?invert_in_darkmode" align=middle width=13.15930604999999pt height=22.831056599999986pt/> is the bias term.

## General deep neural network

The counting of the layer number excludes the input layer. The <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/dc2b6e558ecfe63bafdb6dbd1f0cad16.svg?invert_in_darkmode" align=middle width=56.09580239999998pt height=24.65753399999998pt/>-layer neural network is denoted by
<p align="center"><img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/be8de262e8f505f6d67fae76f77aea72.svg?invert_in_darkmode" align=middle width=629.5055150999999pt height=19.526994300000002pt/></p>
where <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/865e70ab5839636feaab8a6745125c4a.svg?invert_in_darkmode" align=middle width=120.62059019999998pt height=29.190975000000005pt/>, <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/998139a600e0e2203867005393bb05b4.svg?invert_in_darkmode" align=middle width=86.43477479999999pt height=29.190975000000005pt/>, <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/b5142f01744a994ace1bc28b20b87eed.svg?invert_in_darkmode" align=middle width=94.55845409999999pt height=22.831056599999986pt/>, <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/44cbe35529cbf43049034e4ffb71f1bc.svg?invert_in_darkmode" align=middle width=92.96853389999998pt height=22.831056599999986pt/>, <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/8cda31ed38c6d59d14ebefa440099572.svg?invert_in_darkmode" align=middle width=9.98290094999999pt height=14.15524440000002pt/> is a scalar function and ``<img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/c0463eeb4772bfde779c20d52901d01b.svg?invert_in_darkmode" align=middle width=8.219209349999991pt height=14.611911599999981pt/>'' means entry-wise operation. We denote <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/b1d48ece5807e074310a0b4819449aa4.svg?invert_in_darkmode" align=middle width=327.86901014999995pt height=29.190975000000005pt/>. <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/aecfc5cb6de7354f538580fc23ff7eec.svg?invert_in_darkmode" align=middle width=31.472604899999986pt height=34.337843099999986pt/> denotes an entry. This can also be defined recursively,

<p align="center"><img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/a30dad34ccc491270ae2eade2eb3751b.svg?invert_in_darkmode" align=middle width=83.27622600000001pt height=22.127694599999998pt/></p>
<p align="center"><img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/11809237b6942b518865954e3823c66a.svg?invert_in_darkmode" align=middle width=366.48016184999994pt height=22.127694599999998pt/></p>
<p align="center"><img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/0dde269a93e89d8dfa2b02131c7efdef.svg?invert_in_darkmode" align=middle width=393.61014989999995pt height=22.127694599999998pt/></p>

## Complexity

The VC-dimension of a hypothesis class <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/930b956ef51654e0669455a2cdd62fb5.svg?invert_in_darkmode" align=middle width=14.794451099999991pt height=22.55708729999998pt/> is denoted as VCdim(<img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/8209c0f8b3c5233ea2e20dae55588c43.svg?invert_in_darkmode" align=middle width=14.041179899999989pt height=22.465723500000017pt/>).

The Rademacher complexity of a hypothesis space <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/8209c0f8b3c5233ea2e20dae55588c43.svg?invert_in_darkmode" align=middle width=14.041179899999989pt height=22.465723500000017pt/> on a sample set <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/e257acd1ccbe7fcb654708f1a866bfe9.svg?invert_in_darkmode" align=middle width=11.027402099999989pt height=22.465723500000017pt/> is denoted by <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/eb048e4d3123034dac2256effd67ad18.svg?invert_in_darkmode" align=middle width=65.98741049999998pt height=24.65753399999998pt/>.

## Training

The Gradient Descent is oftern denoted by GD. THe Stochastic Gradient Descent is ofter denoted by SGD.

The learning rate is denoted by <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/1d0496971a2775f4887d1df25cea4f7e.svg?invert_in_darkmode" align=middle width=8.751954749999989pt height=14.15524440000002pt/>.

## Gram matrix

The Gram matrix is denoted by <img src="https://rawgit.com/Mayuyu/standard-math-notations-machine-learning/master/svgs/96b697078d351b7b43bd5b5dce0254cd.svg?invert_in_darkmode" align=middle width=22.08723494999999pt height=22.465723500000017pt/>
