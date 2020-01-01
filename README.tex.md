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

Dataset $S=\{\mathbf{z}_i\}^n_{i=1}=\{(\mathbf{x}_i,\mathbf{y}_i)\}^n_{i=1}$ is sampled from a distribution $\mathcal{D}$ over a domain $\mathcal{Z}=\mathcal{X}\times\mathcal{Y}$.

- $\mathcal{X}$ is the instances domain (a set)
- $\mathcal{Y}$ is the label domain (a set)
- $\mathcal{Z}=\mathcal{X}\times\mathcal{Y}$ is the examples domain (a set)

Usually, $\mathcal{X}$ is a subset of $\mathbb{R}^d$ and $\mathcal{Y}$ is a subset of $\mathbb{R}^{d_0}$, where $d$ is the input dimension, $d_0$ is the ouput dimension.

$n=|S|$ is the number of samples. Wihout other specified, $s$ and $n$ are for the training set.

## Function

Hypothesis space is denoted by $\mathcal{H}$. Hypothesis function is denoted by $f_{\mathbf{\theta}}\in\mathcal{H}$ with $f_{\mathbf{\theta}}:\mathcal{X}\to\mathcal{Y}$.

$\mathbf{\theta}$ denotes the set of parameters of $f_{\mathbf{\theta}}$.

The target function is denoted by $f^*:\mathcal{X}\to\mathcal{Y}$ satisfying $\mathbf{y}_i=f^*(\mathbf{x}_i)$ for $i=1,\dots,n$.

## Loss function

Loss function, denoted by $\ell:\mathcal{H}\times\mathcal{Z}\to\mathbb{R}_{+}:=[0,+\infty)$ which measures the difference between a predicted label and a true label, e.g.,

- $L^2$ loss: $\ell(f_{\mathbf{\theta}},\mathbf{z})=(f_{\mathbf{\theta}}(\mathbf{x})-\mathbf{y})^2$, where $\mathbf{z}=(\mathbf{x},\mathbf{y})$. $\ell(f_{\mathbf{\theta}},\mathbf{z})$ can also be written as $\ell(f_{\mathbf{\theta}},\mathbf{y}))$ for convenience.

Empirical risk or training loss is denoted by $L_S(\mathbf{\theta})$ or $\hat{L}_n(\mathbf{\theta})$,

$$
  L_S(\mathbf{\theta})=\frac{1}{n}\sum^n_{i=1}\ell(f_{\mathbf{\theta}}(\mathbf{x}_i),\mathbf{y}_i),
$$

without further explanation $L$ will be used for $L_S$.

The population risk or expected loss is denoted by

$$
  L_{\mathcal{D}}(\mathbf{\theta})=\mathbb{E}_{\mathcal{D}}\ell(f_{\mathbf{\theta}}(\mathbf{x}),\mathbf{y})).
$$

## Activation function

Activation function is denoted by $\sigma(x)$. Some commonly used activation functions are

- $\sigma(z)=\text{ReLU}(z)=\text{max}(0,z)$
- $\sigma(z)=\text{sigmoid}(z)=\dfrac{1}{1+e^{-z}}$
- $\sigma(z)=\tanh(z)$
- $\sigma(z)=\cos z, \sin z$

## Two-layer neural network

THe neuron number of the hidden layer is denoted by $m$,

$$
  f_{\mathbf{\theta}}(\mathbf{x})=\sum^m_{j=1}a_j\sigma(\mathbf{w}_j\cdot\mathbf{x}+b_j),
$$

where $\sigma$ is the activation function, $\mathbf{w}_j$ is the input weight, $a_j$ is the output weight, $b_j$ is the bias term.

## General deep neural network

The counting of the layer number excludes the input layer. The $(H+1)$-layer neural network is denoted by
$$
  f_{\mathbf{\theta}}(\mathbf{x})=\mathbf{W}^{[H]}\sigma\circ(\mathbf{W}^{[H-1]})\sigma\circ(\dots(\mathbf{W}^{[1]}\sigma\circ(\mathbf{W}^{[0]}\mathbf{x}+\mathbf{b}^{[0]})+\mathbf{b}^{[1]})\dots)+\mathbf{b}^{[H-1]})+\mathbf{b}^{[H]},
$$

where $\mathbf{W}^{[l]}\in\mathbb{R}^{m_{l+1}\times m_l}$, $\mathbf{b}^{[l]}=\mathbb{R}^{m_{l+1}}$, $m_0=d_\text{in}=d$, $m_{H+1}=d_\text{out}$, $\sigma$ is a scalar function and ``$\circ$'' means entry-wise operation. We denote $\mathbf{\theta}=(\mathbf{W}^{[0]},\mathbf{W}^{[1]},\dots,\mathbf{W}^{[H]},\mathbf{b}^{[0]},\mathbf{b}^{[1]},\dots,\mathbf{b}^{[H]})$. $\mathbf{W}^{[l]}_{ij}$ denotes an entry. This can also be defined recursively,

$$
  f^{[0]}_{\mathbf{\theta}}(\mathbf{x})=x,
$$
$$
  f^{[l]}_{\mathbf{\theta}}(\mathbf{x})=\sigma\circ(\mathbf{W}^{[l-1]}f^{[l-1]}_{\mathbf{\theta}}(\mathbf{x})+\mathbf{b}^{[H]}), \quad 1\le l\le H,
$$
$$
  f_{\mathbf{\theta}}(\mathbf{x})=f^{[H+1]}_{\mathbf{\theta}}(\mathbf{x})=\mathbf{W}^{[H]}f^{[H]}_{\mathbf{\theta}}(\mathbf{x})+\mathbf{b}^{[H]}, \quad 1\le l\le H.
$$

## Complexity

The VC-dimension of a hypothesis class $\mathbf{H}$ is denoted as VCdim($\mathcal{H}$).

The Rademacher complexity of a hypothesis space $\mathcal{H}$ on a sample set $S$ is denoted by $R(\mathcal{H}\circ S)$.

## Training

The Gradient Descent is oftern denoted by GD. THe Stochastic Gradient Descent is ofter denoted by SGD.

The learning rate is denoted by $\eta$.

## Gram matrix

The Gram matrix is denoted by $K_n$
