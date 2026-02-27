# Chapter 7: Neural Network Architectures

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain the Universal Approximation Theorem and its implications for the expressive power of multilayer perceptrons.
2. Describe the convolution operation and its variants, and trace the evolution of CNN architectures from LeNet to EfficientNet.
3. Analyze the vanishing gradient problem in vanilla RNNs and explain how LSTM and GRU architectures address it through gating mechanisms.
4. Derive the attention mechanism from first principles, from sequence-to-sequence models to scaled dot-product attention.
5. Walk through the complete Transformer architecture, including positional encoding, multi-head attention, feed-forward networks, and residual connections.
6. Explain Vision Transformers (ViT) and their key design decisions, including patch embedding and positional encoding strategies.
7. Describe State Space Models (S4, Mamba) and their advantages for processing long sequences.
8. Understand Mixture of Experts (MoE) architectures and their role in scaling models efficiently.
9. Explain the RWKV architecture and how it achieves linear complexity for sequence modeling.
10. Apply architecture design principles regarding depth, width, normalization, and residual connections.

---

## 7.1 Multilayer Perceptrons (MLPs)

The multilayer perceptron is the simplest form of deep neural network: a stack of fully connected (dense) layers with nonlinear activation functions. Despite their simplicity, MLPs are powerful function approximators and remain essential building blocks within more complex architectures.

### 7.1.1 The Universal Approximation Theorem

The Universal Approximation Theorem (Cybenko, 1989; Hornik et al., 1989) states that a feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on a compact subset of $\mathbb{R}^n$ to arbitrary precision, provided the activation function is non-constant, bounded, and monotonically increasing (or more generally, any non-polynomial activation).

Formally, for any continuous function $f: [0,1]^n \to \mathbb{R}$ and any $\epsilon > 0$, there exists a network:

$$F(\mathbf{x}) = \sum_{j=1}^{N} \alpha_j \sigma(\mathbf{w}_j^T \mathbf{x} + b_j)$$

such that $|F(\mathbf{x}) - f(\mathbf{x})| < \epsilon$ for all $\mathbf{x} \in [0,1]^n$.

The theorem is an **existence** result: it guarantees that a sufficiently wide network can represent any continuous function, but says nothing about how to find the right weights or how many neurons are needed. In practice, deeper networks often generalize better than wider ones for the same parameter count, a phenomenon we discuss in Section 7.12.

### 7.1.2 Depth vs. Width

While a single hidden layer suffices in theory, deep networks are exponentially more efficient than shallow ones for representing certain function classes (Telgarsky, 2016). A network with $L$ layers and width $w$ can represent functions that require a single-layer network of width $O(w^L)$. This "depth efficiency" is one of the key insights motivating deep learning.

### 7.1.3 Activation Functions

Activation functions introduce nonlinearity. Without them, any composition of linear layers collapses to a single linear transformation. The choice of activation function significantly affects training dynamics.

**ReLU (Rectified Linear Unit)** (Nair & Hinton, 2010):

$$\text{ReLU}(x) = \max(0, x)$$

Advantages: computationally efficient, does not saturate for positive inputs, alleviates the vanishing gradient problem. Disadvantage: "dying ReLU" --- neurons with persistently negative inputs have zero gradient and never recover.

**GELU (Gaussian Error Linear Unit)** (Hendrycks & Gimpel, 2016):

$$\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{2/\pi}(x + 0.044715x^3)\right]\right)$$

where $\Phi$ is the CDF of the standard normal distribution. GELU is smooth, non-monotonic near zero, and is the default activation in Transformers (BERT, GPT).

**SiLU / Swish** (Ramachandran et al., 2017):

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

where $\sigma$ is the sigmoid function. Discovered by neural architecture search. Smooth, non-monotonic, self-gated. Used in EfficientNet and many modern architectures.

**Mish** (Misra, 2019):

$$\text{Mish}(x) = x \cdot \tanh(\text{softplus}(x)) = x \cdot \tanh(\ln(1 + e^x))$$

Similar properties to SiLU but with a slightly different curve. Used in YOLOv4 and other detection architectures.

```python
import torch
import torch.nn as nn

# All available as PyTorch modules
activations = {
    'relu': nn.ReLU(),
    'gelu': nn.GELU(),
    'silu': nn.SiLU(),         # Also known as Swish
    'mish': nn.Mish(),
}
```

---

## 7.2 Convolutional Neural Networks (CNNs)

Convolutional Neural Networks exploit the spatial structure of data (images, audio, time series) through the convolution operation, which provides translation equivariance and parameter sharing (LeCun et al., 1998).

### 7.2.1 The Convolution Operation

In the discrete case relevant to neural networks, the 2D convolution of an input feature map $\mathbf{X}$ with a kernel $\mathbf{K}$ of size $k \times k$ produces an output:

$$(X * K)[i, j] = \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} X[i+m, j+n] \cdot K[m, n]$$

Strictly speaking, this is **cross-correlation** (the kernel is not flipped), but the deep learning community calls it convolution. Key parameters include:

**Stride**: The step size of the kernel. Stride $s > 1$ downsamples the output. For input size $H$, kernel size $k$, padding $p$, and stride $s$, the output size is:

$$H_{\text{out}} = \left\lfloor\frac{H - k + 2p}{s}\right\rfloor + 1$$

**Padding**: Zeros added around the input to control the output size. "Same" padding ($p = \lfloor k/2 \rfloor$) preserves the spatial dimensions.

**Dilation**: Spaces between kernel elements. A dilation rate of $d$ means the effective kernel size is $k + (k-1)(d-1)$. Dilated convolutions expand the receptive field without increasing the number of parameters.

```python
import torch.nn as nn

# Standard convolution: 3 input channels, 64 output channels, 3x3 kernel
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
                 stride=1, padding=1, dilation=1, bias=True)
# Input: (batch, 3, H, W) -> Output: (batch, 64, H, W) with padding='same'
```

### 7.2.2 Pooling Operations

Pooling reduces spatial dimensions and provides a degree of translation invariance:

- **Max pooling**: Takes the maximum value in each window. Preserves the strongest activations.
- **Average pooling**: Takes the mean value. Smoother downsampling.
- **Global Average Pooling (GAP)**: Reduces each feature map to a single value by averaging over all spatial positions. Commonly used to replace fully connected layers at the end of CNNs, reducing parameters and overfitting (Lin et al., 2014).

```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)            # Halves spatial dims
gap = nn.AdaptiveAvgPool2d(1)                             # Global average pooling
```

### 7.2.3 1x1 Convolutions

A $1 \times 1$ convolution applies a linear transformation across channels at each spatial position. It is equivalent to a fully connected layer applied independently at every pixel. Uses include:

- **Channel reduction/expansion**: Reducing the number of channels before an expensive $3 \times 3$ convolution (as in Inception/GoogLeNet).
- **Adding nonlinearity**: Combined with activation functions, 1x1 convolutions add cross-channel nonlinear interactions without changing spatial dimensions.
- **Pointwise mixing**: In modern architectures like MobileNet and EfficientNet.

### 7.2.4 Depthwise Separable Convolutions

Standard convolution with $C_{\text{in}}$ input channels, $C_{\text{out}}$ output channels, and kernel size $k$ has $C_{\text{in}} \cdot C_{\text{out}} \cdot k^2$ parameters. Depthwise separable convolutions (Howard et al., 2017) factorize this into:

1. **Depthwise convolution**: A separate $k \times k$ filter per input channel ($C_{\text{in}} \cdot k^2$ parameters).
2. **Pointwise convolution**: A $1 \times 1$ convolution mixing channels ($C_{\text{in}} \cdot C_{\text{out}}$ parameters).

Total parameters: $C_{\text{in}}(k^2 + C_{\text{out}})$, which is approximately $k^2$ times fewer than standard convolution. This factorization is the foundation of MobileNet and EfficientNet.

---

## 7.3 Landmark CNN Architectures

The evolution of CNN architectures over the past three decades illustrates key design principles: depth, skip connections, efficient computation, and principled scaling.

### 7.3.1 LeNet-5 (1998)

LeNet-5 (LeCun et al., 1998) was one of the first successful CNNs, designed for handwritten digit recognition on the MNIST dataset. Its architecture follows a pattern that remains relevant today:

- **Conv (5x5, 6 filters)** -> Average Pool (2x2) -> **Conv (5x5, 16 filters)** -> Average Pool (2x2) -> **FC (120)** -> **FC (84)** -> **FC (10)**

Total parameters: approximately 60,000. LeNet-5 demonstrated two fundamental principles: (1) learned convolutional features outperform hand-crafted features, and (2) alternating convolution and pooling layers create a hierarchical feature extraction pipeline where early layers detect edges, middle layers detect parts, and later layers detect objects.

```python
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)
```

### 7.3.2 AlexNet (2012)

AlexNet (Krizhevsky et al., 2012) won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) by a dramatic margin, reducing the top-5 error rate from 26.2% to 16.4% and catalyzing the deep learning revolution. Key innovations:

- **ReLU activation** (instead of sigmoid/tanh), enabling training of deeper networks by alleviating vanishing gradients. This was the first large-scale demonstration that ReLU was superior.
- **Dropout** (p=0.5) in the fully connected layers for regularization.
- **Data augmentation** (random crops from 256x256 to 224x224, horizontal flips, PCA-based color jittering).
- **GPU training** split across two NVIDIA GTX 580 GPUs, necessitating a split architecture where each GPU processed half the channels.
- **Local Response Normalization (LRN)**, which was later superseded by Batch Normalization.

Architecture: 5 convolutional layers + 3 fully connected layers, ~60M parameters. The first layer used large 11x11 kernels with stride 4, aggressively downsampling the 224x224 input.

### 7.3.3 VGGNet (2014)

VGGNet (Simonyan & Zisserman, 2015) demonstrated that depth matters: using only $3 \times 3$ convolutions stacked deeply (16 or 19 layers) achieved strong performance. Two stacked $3 \times 3$ convolutions have the same receptive field as a single $5 \times 5$ but with fewer parameters and more nonlinearity.

### 7.3.4 GoogLeNet / Inception (2014)

GoogLeNet (Szegedy et al., 2015) introduced the **Inception module**: parallel branches with $1 \times 1$, $3 \times 3$, $5 \times 5$ convolutions and $3 \times 3$ max pooling, concatenated along the channel dimension. The $1 \times 1$ convolutions before the larger kernels reduce computational cost. GoogLeNet achieved top performance with only ~6.8M parameters (compared to VGG's ~138M).

### 7.3.5 ResNet (2015)

ResNet (He et al., 2016) is arguably the most influential CNN architecture. It introduced **residual connections** (skip connections) that enabled training networks of unprecedented depth (up to 152 layers, later 1000+).

**The Problem**: Very deep networks suffer from degradation --- adding more layers increases training error (not just test error). This is distinct from vanishing gradients; it is an optimization difficulty.

**The Solution**: Instead of learning a direct mapping $\mathcal{H}(\mathbf{x})$, each block learns a residual:

$$\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x}$$

The output of the block is:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$$

**Why Residual Connections Work**: Consider the gradient flow. For a network with $L$ residual blocks, the output is:

$$\mathbf{x}_L = \mathbf{x}_0 + \sum_{i=0}^{L-1} \mathcal{F}_i(\mathbf{x}_i)$$

The gradient of the loss with respect to an early layer $\mathbf{x}_l$ is:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \cdot \frac{\partial \mathbf{x}_L}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \left(1 + \frac{\partial}{\partial \mathbf{x}_l}\sum_{i=l}^{L-1}\mathcal{F}_i(\mathbf{x}_i)\right)$$

The key insight is the **additive 1**: even if the residual gradients $\frac{\partial}{\partial \mathbf{x}_l}\sum \mathcal{F}_i$ vanish, the gradient signal is still propagated through the identity path. This provides a "gradient highway" that bypasses the nonlinear transformations, enabling effective training of very deep networks.

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual                      # Skip connection
        out = torch.relu(out)
        return out
```

When the input and output dimensions differ (e.g., during downsampling), a **projection shortcut** (1x1 convolution) is used.

### 7.3.6 DenseNet (2017)

DenseNet (Huang et al., 2017) extends the residual connection idea: each layer receives inputs from **all** preceding layers via concatenation (not addition). For a layer $l$ receiving feature maps from all layers $0, 1, \ldots, l-1$:

$$\mathbf{x}_l = H_l([\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_{l-1}])$$

This encourages feature reuse, reduces the number of parameters (narrower layers suffice), and strengthens gradient flow.

### 7.3.7 EfficientNet (2019)

EfficientNet (Tan & Le, 2019) introduced **compound scaling**: instead of arbitrarily increasing depth, width, or resolution, all three are scaled together using a compound coefficient $\phi$:

$$\text{depth}: d = \alpha^\phi, \quad \text{width}: w = \beta^\phi, \quad \text{resolution}: r = \gamma^\phi$$

subject to $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ (to approximately double the computational cost with each increment of $\phi$). The base architecture (EfficientNet-B0) was found via neural architecture search and uses mobile inverted bottleneck blocks (MBConv) with depthwise separable convolutions and squeeze-and-excitation modules.

---

## 7.4 Recurrent Neural Networks (RNNs)

Recurrent Neural Networks process sequential data by maintaining a hidden state that evolves over time. At each time step $t$, the network takes an input $\mathbf{x}_t$ and the previous hidden state $\mathbf{h}_{t-1}$, and produces a new hidden state and output.

### 7.4.1 Vanilla RNN

The simplest RNN computes:

$$\mathbf{h}_t = \tanh(\mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{W}_{xh}\mathbf{x}_t + \mathbf{b}_h)$$
$$\mathbf{y}_t = \mathbf{W}_{hy}\mathbf{h}_t + \mathbf{b}_y$$

### 7.4.2 The Vanishing Gradient Problem

The gradient of the loss at time $T$ with respect to the hidden state at time $t$ involves:

$$\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} = \prod_{k=t+1}^{T} \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} = \prod_{k=t+1}^{T} \text{diag}(\tanh'(\mathbf{z}_k)) \cdot \mathbf{W}_{hh}$$

where $\mathbf{z}_k = \mathbf{W}_{hh}\mathbf{h}_{k-1} + \mathbf{W}_{xh}\mathbf{x}_k + \mathbf{b}_h$.

Since $\tanh'(z) \leq 1$ and this product involves $T - t$ matrix multiplications, the gradient either:
- **Vanishes** ($\to 0$) when the spectral radius of $\mathbf{W}_{hh}$ is less than 1: long-range dependencies are not learned.
- **Explodes** ($\to \infty$) when the spectral radius exceeds 1: training becomes unstable.

This was formally analyzed by Bengio et al. (1994) and is the fundamental limitation of vanilla RNNs. Gradient clipping addresses the exploding gradient problem but cannot fix vanishing gradients.

### 7.4.3 Long Short-Term Memory (LSTM)

The LSTM (Hochreiter & Schmidhuber, 1997) solves the vanishing gradient problem by introducing a **cell state** $\mathbf{c}_t$ that acts as a "conveyor belt" for information, protected by three gates that control what information enters, exits, and is retained.

**Forget Gate** --- decides what to discard from the cell state:
$$\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$

**Input Gate** --- decides what new information to store:
$$\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$

**Candidate Cell State** --- creates a vector of new candidate values:
$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$

**Cell State Update** --- combines forgotten old information with new candidates:
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

**Output Gate** --- decides what to output based on the cell state:
$$\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

where $\odot$ denotes element-wise multiplication and $\sigma$ is the sigmoid function.

The critical insight is that the cell state update $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$ is an **additive** interaction. The gradient of $\mathbf{c}_T$ with respect to $\mathbf{c}_t$ involves products of forget gate values $\prod_{k=t+1}^{T} \mathbf{f}_k$, which can stay close to 1 when the forget gates are close to 1, allowing gradients to flow over long distances.

```python
# PyTorch LSTM
lstm = nn.LSTM(
    input_size=128,
    hidden_size=256,
    num_layers=2,         # Stacked LSTM
    batch_first=True,
    dropout=0.2,          # Between stacked layers (not after last)
    bidirectional=True    # Process sequence in both directions
)

# Input: (batch, seq_len, input_size)
# Output: (batch, seq_len, 2 * hidden_size) for bidirectional
output, (h_n, c_n) = lstm(input_seq)
```

### 7.4.4 Gated Recurrent Unit (GRU)

The GRU (Cho et al., 2014) simplifies the LSTM by combining the forget and input gates into a single **update gate** and merging the cell state and hidden state:

**Update Gate** --- controls how much of the past to retain:
$$\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_z)$$

**Reset Gate** --- controls how much of the past to forget when computing the candidate:
$$\mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r)$$

**Candidate Hidden State**:
$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_h)$$

**Hidden State Update**:
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

The GRU has fewer parameters than the LSTM (two gates vs. three, no separate cell state) and performs comparably on many tasks.

### 7.4.5 Bidirectional and Stacked RNNs

**Bidirectional RNNs** process the sequence in both forward and backward directions, concatenating the hidden states. This allows each position to have context from both past and future tokens.

**Stacked RNNs** use multiple layers, where the output of one layer becomes the input to the next. Stacking increases the model's capacity to learn hierarchical representations.

---

## 7.5 The Attention Mechanism

Attention is the mechanism that enables neural networks to focus on relevant parts of the input when producing each part of the output. It is the most important innovation in modern deep learning.

### 7.5.1 Sequence-to-Sequence with Attention (Bahdanau)

The attention mechanism was introduced by Bahdanau et al. (2015) to address a bottleneck in sequence-to-sequence models: compressing the entire input into a single fixed-length vector.

In the standard encoder-decoder framework:
- The **encoder** (typically a bidirectional RNN) produces a sequence of hidden states $\mathbf{h}_1, \mathbf{h}_2, \ldots, \mathbf{h}_T$.
- The **decoder** generates output tokens one at a time, using a context vector at each step.

Without attention, the context vector is just the final encoder hidden state. With Bahdanau attention, the context vector $\mathbf{c}_t$ at decoder step $t$ is a weighted sum of all encoder hidden states:

$$\mathbf{c}_t = \sum_{j=1}^{T} \alpha_{tj} \mathbf{h}_j$$

where the attention weights $\alpha_{tj}$ indicate how much attention decoder step $t$ pays to encoder position $j$:

$$\alpha_{tj} = \frac{\exp(e_{tj})}{\sum_{k=1}^{T} \exp(e_{tk})}$$

The energy scores $e_{tj}$ are computed by an alignment function:

$$e_{tj} = \mathbf{v}^T \tanh(\mathbf{W}_s \mathbf{s}_{t-1} + \mathbf{W}_h \mathbf{h}_j)$$

where $\mathbf{s}_{t-1}$ is the previous decoder state, and $\mathbf{v}$, $\mathbf{W}_s$, $\mathbf{W}_h$ are learned parameters. This is **additive attention**.

### 7.5.2 Luong Attention

Luong et al. (2015) proposed a simpler **multiplicative attention**:

$$e_{tj} = \mathbf{s}_t^T \mathbf{W}_a \mathbf{h}_j$$

or even simpler, **dot-product attention**:

$$e_{tj} = \mathbf{s}_t^T \mathbf{h}_j$$

### 7.5.3 Scaled Dot-Product Attention (Vaswani)

The Transformer (Vaswani et al., 2017) generalized attention by introducing the **Query-Key-Value** framework. Given:
- **Queries** $\mathbf{Q} \in \mathbb{R}^{n \times d_k}$: what we are looking for
- **Keys** $\mathbf{K} \in \mathbb{R}^{m \times d_k}$: what we compare against
- **Values** $\mathbf{V} \in \mathbb{R}^{m \times d_v}$: the information we retrieve

Scaled dot-product attention computes:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

The scaling factor $\sqrt{d_k}$ prevents the dot products from growing too large in magnitude. When $d_k$ is large, the dot products can have large variance, pushing the softmax into regions with extremely small gradients (saturation). Scaling by $\sqrt{d_k}$ keeps the variance approximately 1, assuming the query and key components have zero mean and unit variance.

### 7.5.4 Multi-Head Attention

Rather than performing a single attention function with $d_{\text{model}}$-dimensional keys, values, and queries, multi-head attention linearly projects them $h$ times with different learned projections, performs attention in parallel, and concatenates the results:

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}^O$$

where each head is:

$$\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)$$

with projection matrices $\mathbf{W}_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $\mathbf{W}_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $\mathbf{W}_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$, and $\mathbf{W}^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$.

Typically, $d_k = d_v = d_{\text{model}} / h$. With $h = 8$ heads and $d_{\text{model}} = 512$, each head operates on $d_k = 64$ dimensions. Multi-head attention allows the model to attend to information from different representation subspaces at different positions simultaneously.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections and reshape to (batch, n_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)

        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * self.d_k)
        return self.W_o(context)
```

---

## 7.6 The Transformer

The Transformer (Vaswani et al., 2017) eliminated recurrence entirely, processing entire sequences in parallel through self-attention. It is the foundation of virtually all modern language models (BERT, GPT, T5) and has been adapted to vision (ViT), audio, and multimodal tasks.

### 7.6.1 High-Level Architecture

The Transformer consists of an **encoder** and a **decoder**, each composed of stacked identical blocks. The original paper used 6 blocks each. However, many modern applications use only the encoder (BERT for classification) or only the decoder (GPT for generation).

### 7.6.2 Input Embedding and Positional Encoding

Since the Transformer has no recurrence or convolution, it has no inherent notion of position. **Positional encoding** injects position information into the input embeddings.

**Input Embedding**: Each input token $x_i$ is mapped to a dense vector $\mathbf{e}_i \in \mathbb{R}^{d_{\text{model}}}$ via a learned embedding matrix.

**Sinusoidal Positional Encoding** (Vaswani et al., 2017):

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

where $pos$ is the position and $i$ is the dimension index. Each dimension corresponds to a sinusoid with a different wavelength, ranging from $2\pi$ to $10000 \cdot 2\pi$. This encoding has the property that $PE_{pos+k}$ can be expressed as a linear function of $PE_{pos}$, allowing the model to learn to attend to relative positions.

**Learned Positional Encoding**: An alternative that simply learns a separate embedding for each position. Used in BERT and GPT-2. Equivalent in practice for fixed-length inputs but does not generalize to unseen positions.

The final input to the Transformer is $\mathbf{e}_i + PE_i$ (element-wise addition, not concatenation).

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)       # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

### 7.6.3 Encoder Block

Each encoder block consists of two sub-layers:

1. **Multi-Head Self-Attention**: The input sequence attends to itself. Q, K, and V are all derived from the same input.
2. **Position-wise Feed-Forward Network (FFN)**: Two linear transformations with a nonlinear activation:

$$\text{FFN}(\mathbf{x}) = \text{GELU}(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

where $\mathbf{W}_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$ and $\mathbf{W}_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$. Typically $d_{ff} = 4 \times d_{\text{model}}$. The original paper used ReLU; modern implementations use GELU.

Each sub-layer is wrapped with a **residual connection** and **layer normalization**:

$$\text{output} = \text{LayerNorm}(\mathbf{x} + \text{SubLayer}(\mathbf{x}))$$

```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Post-norm (original Transformer)
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x
```

### 7.6.4 Decoder Block

Each decoder block has three sub-layers:

1. **Masked Multi-Head Self-Attention**: The decoder attends to itself, but future positions are masked to prevent information leakage during autoregressive generation. This is implemented by setting attention scores for future positions to $-\infty$ before the softmax.

2. **Multi-Head Cross-Attention**: The decoder attends to the encoder output. Q comes from the decoder, while K and V come from the encoder.

3. **Position-wise FFN**: Same as the encoder.

Each sub-layer has residual connections and layer normalization.

```python
class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention (causal mask prevents future peeking)
        attn = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn))

        # Cross-attention to encoder output
        attn = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn))

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        return x
```

### 7.6.5 Output Projection

The decoder's final output is projected to a vocabulary-sized vector via a linear layer, followed by softmax to produce token probabilities:

$$P(y_t | y_{<t}, \mathbf{x}) = \text{softmax}(\mathbf{W}_{\text{vocab}} \mathbf{h}_t + \mathbf{b}_{\text{vocab}})$$

A common practice is **weight tying**: sharing the weights between the input embedding matrix and the output projection matrix (Press & Wolf, 2017).

### 7.6.6 Complete Transformer

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, n_heads=8,
                 n_encoder_layers=6, n_decoder_layers=6, d_ff=2048,
                 max_len=5000, dropout=0.1):
        super().__init__()

        self.src_embedding = nn.Embedding(src_vocab, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])

        self.output_proj = nn.Linear(d_model, tgt_vocab)
        self.d_model = d_model

    def encode(self, src, src_mask=None):
        x = self.pos_encoding(self.src_embedding(src) * (self.d_model ** 0.5))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, memory, src_mask=None, tgt_mask=None):
        x = self.pos_encoding(self.tgt_embedding(tgt) * (self.d_model ** 0.5))
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.output_proj(output)
```

### 7.6.7 Computational Complexity of Self-Attention

Self-attention has $O(n^2 \cdot d)$ time and $O(n^2)$ memory complexity, where $n$ is the sequence length and $d$ is the model dimension. To understand why: the $\mathbf{Q}\mathbf{K}^T$ computation produces an $n \times n$ matrix, which must be stored for the softmax and the subsequent multiplication with $\mathbf{V}$. For $n = 4096$ and $d = 512$, the attention matrix alone requires $4096^2 \times 4 = 67$ MB per layer per head in float32.

This quadratic scaling in sequence length is the primary limitation of the standard Transformer and has motivated extensive research into efficient attention mechanisms (Tay et al., 2022):

- **FlashAttention** (Dao et al., 2022): Does not reduce computational complexity but avoids materializing the full $n \times n$ attention matrix by using tiling and recomputation, achieving significant wall-clock speedups and memory savings.
- **Linear attention**: Approximates softmax attention with kernel feature maps, reducing complexity to $O(n \cdot d^2)$.
- **Sparse attention** (Longformer, BigBird): Restricts attention to local windows and selected global tokens.
- **State Space Models** (Section 7.8): Replace attention entirely with linear recurrences.

In practice, FlashAttention has become the standard implementation for Transformer training, as it provides substantial speedups (2-4x) without approximation.

---

## 7.7 Vision Transformers (ViT)

The Vision Transformer (Dosovitskiy et al., 2021) demonstrated that a pure Transformer, with minimal modifications, can achieve state-of-the-art image classification when trained on sufficient data.

### 7.7.1 Patch Embedding

ViT divides an image into fixed-size patches and treats each patch as a "token":

1. An image of size $H \times W \times C$ is divided into $N = HW/P^2$ patches of size $P \times P \times C$.
2. Each patch is flattened to a vector of dimension $P^2 \cdot C$.
3. A linear projection maps each flattened patch to $d_{\text{model}}$ dimensions.

This is equivalent to a convolution with kernel size $P$ and stride $P$.

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=768):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (batch, channels, H, W)
        x = self.proj(x)              # (batch, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)   # (batch, n_patches, embed_dim)
        return x
```

### 7.7.2 Class Token and Positional Encoding

Following BERT, ViT prepends a learnable **class token** `[CLS]` to the sequence of patch embeddings. The final representation of this token is used for classification.

Positional encodings are added to each patch embedding (including the class token). ViT uses **learned positional embeddings** rather than sinusoidal ones.

```python
class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 n_classes=1000, embed_dim=768, depth=12, n_heads=12,
                 d_ff=3072, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size,
                                          in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, n_heads, d_ff, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_output = x[:, 0]        # Use [CLS] token for classification
        return self.head(cls_output)
```

### 7.7.3 ViT vs. CNNs

ViT lacks the inductive biases of CNNs (locality, translation equivariance), requiring more data to achieve comparable performance. However, with sufficient data (JFT-300M, ImageNet-21k), ViT matches or exceeds CNNs. Data-efficient variants like **DeiT** (Touvron et al., 2021) use knowledge distillation and strong data augmentation to train ViT effectively on ImageNet alone.

### 7.7.4 Swin Transformer

The Swin Transformer (Liu et al., 2021) introduces a hierarchical structure with **shifted windows**:

1. Self-attention is computed within non-overlapping local windows (e.g., $7 \times 7$ patches), reducing complexity from $O(n^2)$ to $O(n \cdot w^2)$ where $w$ is the window size.
2. Between consecutive layers, windows are shifted by half the window size, enabling cross-window information flow.
3. Patch merging layers progressively reduce spatial resolution while increasing channel dimension, creating a hierarchical feature pyramid (similar to CNNs).

Swin Transformer achieves state-of-the-art results on image classification, object detection, and semantic segmentation, serving as a general-purpose backbone for vision tasks.

---

## 7.8 State Space Models

State Space Models (SSMs) offer an alternative to Transformers for sequence modeling, achieving linear or near-linear complexity in sequence length while maintaining strong performance on long-range dependencies.

### 7.8.1 The State Space Framework

A continuous-time linear state space model maps an input signal $u(t) \in \mathbb{R}$ to an output $y(t) \in \mathbb{R}$ through a hidden state $\mathbf{x}(t) \in \mathbb{R}^N$:

$$\mathbf{x}'(t) = \mathbf{A}\mathbf{x}(t) + \mathbf{B}u(t)$$
$$y(t) = \mathbf{C}\mathbf{x}(t) + Du(t)$$

where $\mathbf{A} \in \mathbb{R}^{N \times N}$, $\mathbf{B} \in \mathbb{R}^{N \times 1}$, $\mathbf{C} \in \mathbb{R}^{1 \times N}$, and $D \in \mathbb{R}$.

For discrete-time inputs (as in deep learning), the continuous system is discretized using a step size $\Delta$:

$$\bar{\mathbf{A}} = \exp(\Delta \mathbf{A}), \quad \bar{\mathbf{B}} = (\Delta \mathbf{A})^{-1}(\exp(\Delta \mathbf{A}) - \mathbf{I}) \cdot \Delta \mathbf{B}$$

$$\mathbf{x}_k = \bar{\mathbf{A}}\mathbf{x}_{k-1} + \bar{\mathbf{B}}u_k$$
$$y_k = \mathbf{C}\mathbf{x}_k$$

### 7.8.2 S4: Structured State Spaces for Sequence Modeling

The S4 model (Gu et al., 2022) made SSMs practical for deep learning through two key innovations:

1. **HiPPO initialization**: The $\mathbf{A}$ matrix is initialized using the HiPPO (High-order Polynomial Projection Operators) framework, which provides a principled way to compress continuous-time history into a fixed-size state.

2. **Efficient computation**: The discrete SSM can be computed as a convolution:

$$y_k = \sum_{j=0}^{k} \bar{\mathbf{C}} \bar{\mathbf{A}}^{k-j} \bar{\mathbf{B}} \cdot u_j = (\bar{K} * u)_k$$

where $\bar{K} = (\bar{\mathbf{C}}\bar{\mathbf{B}}, \bar{\mathbf{C}}\bar{\mathbf{A}}\bar{\mathbf{B}}, \ldots, \bar{\mathbf{C}}\bar{\mathbf{A}}^{L-1}\bar{\mathbf{B}})$ is the SSM kernel. This convolution can be computed in $O(L \log L)$ using FFT during training, while inference uses the recurrent form for $O(1)$ per step.

### 7.8.3 Mamba: Selective State Spaces

Mamba (Gu & Dao, 2023) addresses a fundamental limitation of S4: the state space parameters ($\mathbf{A}$, $\mathbf{B}$, $\mathbf{C}$, $\Delta$) are fixed for all inputs, making the model **linear time-invariant (LTI)**. LTI systems cannot perform content-based reasoning (e.g., "copy this token when you see that token").

Mamba makes the SSM parameters **input-dependent** (selective):

$$\mathbf{B}_k = f_B(\mathbf{x}_k), \quad \mathbf{C}_k = f_C(\mathbf{x}_k), \quad \Delta_k = f_\Delta(\mathbf{x}_k)$$

where $f_B$, $f_C$, $f_\Delta$ are learned projections of the input. This selectivity allows the model to selectively retain or forget information based on the current input, similar to gating in LSTMs.

The key challenge is that input-dependent parameters break the convolution view, preventing FFT-based computation. Mamba addresses this through a **hardware-aware parallel scan algorithm** that computes the recurrence efficiently on GPUs, achieving $O(L)$ time complexity with efficient memory access patterns.

**Advantages of Mamba**:
- **Linear complexity**: $O(L)$ vs. $O(L^2)$ for Transformers.
- **Constant memory per step at inference**: Unlike Transformers whose KV cache grows linearly with sequence length.
- **Strong performance**: Matches or exceeds Transformer-based models on language, audio, and genomics tasks.

### 7.8.4 SSMs vs. Transformers

| Property | Transformer | SSM (Mamba) |
|----------|------------|-------------|
| Training complexity | $O(L^2 d)$ | $O(L d N)$ |
| Inference per step | $O(L d)$ (KV cache) | $O(d N)$ (fixed state) |
| Memory (inference) | $O(L d)$ (KV cache grows) | $O(d N)$ (constant) |
| Content-based reasoning | Strong (attention) | Selective SSMs enable this |
| Long-range dependencies | Theoretically unlimited but costly | Efficient for very long sequences |

---

## 7.9 Mixture of Experts (MoE)

Mixture of Experts (Jacobs et al., 1991; Shazeer et al., 2017) is a technique for scaling model capacity without proportionally scaling computation: only a subset of the model's parameters is activated for each input.

### 7.9.1 Architecture

In a typical MoE Transformer, each FFN layer is replaced by a collection of $E$ expert FFN networks and a **gating network** (router) that selects which experts to activate:

$$\text{MoE}(\mathbf{x}) = \sum_{i=1}^{E} g_i(\mathbf{x}) \cdot \text{Expert}_i(\mathbf{x})$$

where $g_i(\mathbf{x})$ is the gating weight for expert $i$.

### 7.9.2 Top-K Gating

The gating network computes:

$$G(\mathbf{x}) = \text{softmax}(\text{TopK}(\mathbf{W}_g \mathbf{x}, k))$$

where $\text{TopK}$ keeps only the top-$k$ values (typically $k = 1$ or $k = 2$) and sets the rest to $-\infty$ before the softmax. This sparse activation ensures that only $k$ out of $E$ experts are computed for each token, keeping the computational cost per token roughly constant regardless of the total number of experts.

### 7.9.3 Load Balancing

A critical challenge in MoE is ensuring that tokens are distributed evenly across experts. Without explicit balancing, the router tends to collapse to always selecting the same few experts (expert collapse), wasting capacity.

**Auxiliary load balancing loss** (Fedus et al., 2022):

$$\mathcal{L}_{\text{balance}} = \alpha \cdot E \sum_{i=1}^{E} f_i \cdot P_i$$

where $f_i$ is the fraction of tokens routed to expert $i$ and $P_i$ is the average router probability for expert $i$. This loss encourages uniform distribution of tokens.

### 7.9.4 Switch Transformer

The Switch Transformer (Fedus et al., 2022) simplified MoE by using **top-1 routing** (each token goes to exactly one expert), which is simpler and more efficient than top-2. Key design decisions:

- Selective precision: experts operate in float32 for stability.
- Expert capacity factor: limits the maximum number of tokens per expert in a batch to prevent imbalance.
- Simplified routing: direct softmax over expert weights with argmax selection.

### 7.9.5 Mixtral Architecture

Mixtral (Jiang et al., 2024) demonstrated that MoE can be applied to strong open-weight language models. Mixtral 8x7B has 46.7B total parameters but uses only 12.9B per token (2 out of 8 experts activated via top-2 routing). It matches or exceeds the performance of Llama 2 70B while being significantly faster at inference.

### 7.9.6 Advantages and Challenges

**Advantages**:
- Scale model capacity without proportional compute increase.
- Experts can specialize in different types of inputs.
- Near-constant inference cost as the model scales.

**Challenges**:
- Load balancing is difficult and critical for performance.
- All expert parameters must reside in memory (or be managed via expert parallelism across devices).
- Training instability: requires careful initialization and auxiliary losses.
- Communication overhead in distributed settings when experts reside on different devices.

---

## 7.10 RWKV

RWKV (Peng et al., 2023) is a hybrid architecture that combines the training parallelism of Transformers with the inference efficiency of RNNs, achieving linear complexity for both training and inference.

### 7.10.1 Key Innovation: Linear Attention

Standard attention requires $O(n^2)$ computation because each token attends to all others. RWKV replaces softmax attention with a linear attention mechanism based on the **WKV (Weighted Key-Value)** operator:

$$\text{wkv}_t = \frac{\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} v_i + e^{u+k_t} v_t}{\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} + e^{u+k_t}}$$

where $w$ is a learnable channel-wise decay factor, $u$ is a learnable bonus for the current position, and $k_i$, $v_i$ are the key and value vectors at position $i$.

The critical property is that this can be computed **recurrently**: at each step $t$, only a fixed-size state needs to be maintained, updated as:

$$a_t = e^{-w} a_{t-1} + e^{k_t} v_t, \quad b_t = e^{-w} b_{t-1} + e^{k_t}$$
$$\text{wkv}_t = a_t / b_t$$

(simplified form; the actual implementation includes the bonus $u$ term and numerical stability considerations).

### 7.10.2 Architecture Components

RWKV alternates two types of blocks:

1. **Time-mixing block** (replaces attention): Uses the WKV operator with learnable interpolation between the current and previous token's key, value, and receptance (analogous to query) vectors.

2. **Channel-mixing block** (replaces FFN): Uses a gated mechanism with learnable interpolation between current and previous tokens.

Both blocks use **token shift** --- linear interpolation between the current token $x_t$ and the previous token $x_{t-1}$ --- as a lightweight mechanism for incorporating temporal context:

$$\tilde{x}_t = \mu \odot x_t + (1 - \mu) \odot x_{t-1}$$

where $\mu$ is a learnable parameter.

### 7.10.3 Advantages

- **Linear complexity**: $O(Ld)$ for both training and inference, where $L$ is sequence length and $d$ is model dimension.
- **Constant memory at inference**: Like an RNN, the state size is fixed regardless of sequence length.
- **Trainable in parallel**: Unlike RNNs, the WKV computation can be parallelized across the sequence dimension during training.
- **No KV cache**: Inference does not require storing key-value pairs for all past tokens.

### 7.10.4 Comparison to Transformers

RWKV trades the expressiveness of full attention (any token can attend to any other token equally) for computational efficiency. The exponential decay $e^{-w}$ means that attention to distant tokens diminishes with distance, which is a reasonable inductive bias for many tasks but may limit performance on tasks requiring precise long-range lookup.

---

## 7.11 Architecture Design Principles

Having surveyed the major architectures, we now distill the key design principles that guide modern neural network architecture.

### 7.11.1 Depth vs. Width

**Depth** (number of layers) provides hierarchical feature abstraction. Each layer can perform a more complex transformation, and deep networks can represent exponentially more complex functions than shallow ones (Telgarsky, 2016). However, very deep networks are harder to train (vanishing/exploding gradients, optimization landscape complexity).

**Width** (hidden dimension) provides capacity within each layer. Wider networks are easier to optimize (the loss landscape is smoother) but are less parameter-efficient than deeper networks.

**Practical guidelines**:
- For a fixed parameter budget, moderate depth with moderate width generally outperforms extreme depth or extreme width.
- Scaling laws research (Kaplan et al., 2020) suggests that compute-optimal models should scale depth and width together: approximately $d_{\text{model}} \propto N^{0.5}$ and depth $\propto N^{0.5}$ for a model with $N$ parameters.

### 7.11.2 Normalization Placement: Pre-Norm vs. Post-Norm

**Post-norm** (original Transformer):
$$\text{output} = \text{LayerNorm}(x + \text{SubLayer}(x))$$

**Pre-norm** (GPT-2 and most modern models):
$$\text{output} = x + \text{SubLayer}(\text{LayerNorm}(x))$$

Pre-norm is more stable to train (the residual path remains unnormalized, preserving gradient flow) and does not require learning rate warmup. Post-norm can achieve slightly better final performance but is harder to train without careful initialization and warmup (Xiong et al., 2020).

Most modern architectures use pre-norm. Some recent work (DeepNorm) proposes alternatives that combine the stability of pre-norm with the performance of post-norm.

### 7.11.3 Residual Connections

Residual connections are ubiquitous in modern architectures. They serve three purposes:

1. **Gradient flow**: Provide a direct path for gradients, enabling training of very deep networks.
2. **Feature reuse**: Allow lower-level features to bypass transformations.
3. **Ensemble effect**: ResNets can be viewed as implicit ensembles of shallower networks (Veit et al., 2016).

**Design rule**: Every computational block should have a residual connection. The input and output dimensions of the block must match (or a projection shortcut must be used).

### 7.11.4 Scaling Rules

Empirical scaling laws (Kaplan et al., 2020; Hoffmann et al., 2022) have revealed remarkably consistent relationships between model size, data size, compute budget, and performance:

$$L(N) \propto N^{-\alpha}$$

where $L$ is the loss, $N$ is the number of parameters, and $\alpha \approx 0.076$ for language models. Similar power laws hold for data scaling and compute scaling.

The **Chinchilla scaling law** (Hoffmann et al., 2022) established that compute-optimal training should scale data and parameters equally: for a given compute budget $C$, the optimal allocation is approximately $N \propto C^{0.5}$ parameters and $D \propto C^{0.5}$ data tokens.

### 7.11.5 Activation Functions in Different Architectures

| Architecture Family | Common Activation |
|--------------------|-------------------|
| Classical CNNs (ResNet) | ReLU |
| Modern CNNs (EfficientNet) | SiLU / Swish |
| Transformers (BERT, GPT) | GELU |
| Detection (YOLO) | Mish, SiLU |
| SSMs (Mamba) | SiLU |

### 7.11.6 Initialization

Proper weight initialization is critical for training stability. The goal is to maintain the variance of activations (and gradients) roughly constant across layers, preventing exponential growth or decay as signals propagate through the network.

- **Xavier/Glorot initialization** (Glorot & Bengio, 2010): $W \sim \mathcal{U}\left(-\sqrt{6/(n_{\text{in}} + n_{\text{out}})}, \sqrt{6/(n_{\text{in}} + n_{\text{out}})}\right)$. Designed for sigmoid/tanh activations. The derivation assumes that the activation function is linear near zero and ensures that $\text{Var}(\text{output}) = \text{Var}(\text{input})$.

- **Kaiming/He initialization** (He et al., 2015): $W \sim \mathcal{N}(0, 2/n_{\text{in}})$. Designed for ReLU activations, which zero out half the inputs. The factor of 2 compensates for this halving. Used in ResNets and CNNs. For layers followed by other activations, the gain factor is adjusted (e.g., $\sqrt{2/(1+a^2)}$ for leaky ReLU with slope $a$).

- **Truncated normal**: $W \sim \mathcal{N}(0, \sigma^2)$ truncated at $\pm 2\sigma$. Used in ViT and many Transformer variants, typically with $\sigma = 0.02$.

- **GPT-style initialization**: Weights of residual layers are scaled by $1/\sqrt{2N}$ where $N$ is the number of residual blocks, preventing the variance from growing with depth (Radford et al., 2019).

```python
# PyTorch initialization utilities
import torch.nn.init as init

# Kaiming (He) initialization for a conv layer
init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='relu')

# Xavier initialization for a linear layer
init.xavier_uniform_(linear.weight)

# Custom initialization for all layers
def init_weights(module):
    if isinstance(module, nn.Linear):
        init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        init.kaiming_normal_(module.weight, mode='fan_out',
                             nonlinearity='relu')

model.apply(init_weights)
```

### 7.11.7 Attention Variants and Efficiency

The attention landscape has expanded significantly beyond the original scaled dot-product formulation:

- **Multi-Query Attention (MQA)** (Shazeer, 2019): Uses a single shared key-value head across all query heads, reducing the KV cache size by a factor of $h$ (number of heads). Provides substantial inference speedups with modest quality loss.

- **Grouped-Query Attention (GQA)** (Ainslie et al., 2023): A middle ground between multi-head and multi-query attention. Groups of query heads share key-value heads. Used in Llama 2 70B and Mistral.

- **Sliding Window Attention**: Each token attends only to a fixed-size local window, reducing complexity to $O(n \cdot w)$ where $w$ is the window size. Long-range information propagates through multiple layers.

- **Rotary Positional Embeddings (RoPE)** (Su et al., 2024): Encodes position information by rotating the query and key vectors, enabling relative position awareness and better length generalization. Used in LLaMA, Mistral, and most modern LLMs.

---

## 7.12 Architecture Comparison Summary

| Architecture | Complexity | Strengths | Weaknesses |
|-------------|-----------|-----------|------------|
| MLP | $O(d^2)$ per layer | Universal approximator, simple | No spatial/temporal inductive bias |
| CNN | $O(k^2 \cdot c^2 \cdot n)$ | Translation equivariance, local features | Limited receptive field, no global context |
| RNN/LSTM | $O(d^2 \cdot L)$ | Sequential processing, variable length | Sequential (slow), vanishing gradients |
| Transformer | $O(L^2 \cdot d)$ | Global attention, parallel training | Quadratic in sequence length |
| SSM (Mamba) | $O(L \cdot d \cdot N)$ | Linear complexity, constant-memory inference | Newer, less ecosystem support |
| MoE | $O(k \cdot d_{\text{ff}} \cdot d)$ per token | Scale capacity cheaply | Load balancing, memory for all experts |
| RWKV | $O(L \cdot d)$ | Linear training and inference | Decaying attention, less expressive |

---

## Exercises

1. **Universal Approximation**: Implement a single-hidden-layer MLP with ReLU activation. Train it to approximate $f(x) = \sin(5x)$ on $[-1, 1]$. How many hidden units are needed to achieve MSE < 0.001? Compare with a 3-layer MLP of the same total parameter count.

2. **CNN Architecture**: Implement a simplified ResNet-18 from scratch in PyTorch. Train it on CIFAR-10. Then remove the skip connections and retrain. Compare the training and validation curves. At what depth does training without skip connections start to degrade?

3. **Attention Visualization**: Implement scaled dot-product attention from scratch. Create a synthetic sequence-to-sequence task (e.g., reverse a sequence of integers). Train a small Transformer and visualize the attention weights. Do the attention patterns match your expectations?

4. **LSTM vs. GRU**: Implement both LSTM and GRU from scratch (without using `nn.LSTM` or `nn.GRU`). Train both on a character-level language modeling task. Compare convergence speed, final performance, and inference speed.

5. **Transformer from Scratch**: Implement the complete Transformer architecture (encoder + decoder) following the description in Section 7.6. Train it on a small machine translation dataset (e.g., English-to-French with the Tatoeba dataset). Experiment with pre-norm vs. post-norm.

6. **Vision Transformer**: Implement ViT from scratch and train it on CIFAR-10. Compare its performance to a ResNet-18 with a similar parameter count. Does ViT catch up when you add data augmentation? At what dataset size does ViT start to outperform the CNN?

7. **Positional Encoding**: Compare sinusoidal and learned positional encodings by training two identical Transformers on a language modeling task, differing only in their positional encoding. Measure perplexity. Do they converge differently? What happens when you test on sequences longer than those seen during training?

8. **Architecture Analysis**: Given a compute budget of $10^{18}$ FLOPs, use the Chinchilla scaling law to determine the optimal model size and dataset size. How many tokens should be in the training set? How many parameters should the model have?

9. **MoE Implementation**: Implement a simple MoE layer with 4 experts and top-2 gating. Add an auxiliary load balancing loss. Train on a classification task and monitor the expert utilization over training. Does the load balancing loss prevent expert collapse?

10. **Efficiency Comparison**: For a sequence of length $L = 4096$ and model dimension $d = 512$, compute the theoretical FLOPs for (a) standard self-attention, (b) a single Mamba layer with state dimension $N = 16$, and (c) a single RWKV time-mixing block. Discuss the practical implications for training and inference.

---

## References

Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. *Proceedings of the 3rd International Conference on Learning Representations*.

Bengio, Y., Simard, P., & Frasconi, P. (1994). Learning Long-Term Dependencies with Gradient Descent is Difficult. *IEEE Transactions on Neural Networks*, 5(2), 157--166.

Cho, K., van Merrienboer, B., Gulcehre, C., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing*, 1724--1734.

Cybenko, G. (1989). Approximation by Superpositions of a Sigmoidal Function. *Mathematics of Control, Signals and Systems*, 2(4), 303--314.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *Proceedings of the 9th International Conference on Learning Representations*.

Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. *Journal of Machine Learning Research*, 23(120), 1--39.

Glorot, X., & Bengio, Y. (2010). Understanding the Difficulty of Training Deep Feedforward Neural Networks. *Proceedings of the 13th International Conference on Artificial Intelligence and Statistics*, 249--256.

Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv preprint arXiv:2312.00752*.

Gu, A., Goel, K., & Re, C. (2022). Efficiently Modeling Long Sequences with Structured State Spaces. *Proceedings of the 10th International Conference on Learning Representations*.

He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. *Proceedings of the IEEE International Conference on Computer Vision*, 1026--1034.

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 770--778.

Hendrycks, D., & Gimpel, K. (2016). Gaussian Error Linear Units (GELUs). *arXiv preprint arXiv:1606.08415*.

Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735--1780.

Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022). Training Compute-Optimal Large Language Models. *Advances in Neural Information Processing Systems*, 35, 30016--30030.

Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer Feedforward Networks are Universal Approximators. *Neural Networks*, 2(5), 359--366.

Howard, A. G., Zhu, M., Chen, B., et al. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. *arXiv preprint arXiv:1704.04861*.

Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017). Densely Connected Convolutional Networks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 4700--4708.

Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive Mixtures of Local Experts. *Neural Computation*, 3(1), 79--87.

Jiang, A. Q., Sablayrolles, A., Roux, A., et al. (2024). Mixtral of Experts. *arXiv preprint arXiv:2401.04088*.

Kaplan, J., McCandlish, S., Henighan, T., et al. (2020). Scaling Laws for Neural Language Models. *arXiv preprint arXiv:2001.08361*.

Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. *Advances in Neural Information Processing Systems*, 25, 1097--1105.

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-Based Learning Applied to Document Recognition. *Proceedings of the IEEE*, 86(11), 2278--2324.

Lin, M., Chen, Q., & Yan, S. (2014). Network In Network. *Proceedings of the 2nd International Conference on Learning Representations*.

Liu, Z., Lin, Y., Cao, Y., et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 10012--10022.

Luong, M.-T., Pham, H., & Manning, C. D. (2015). Effective Approaches to Attention-based Neural Machine Translation. *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing*, 1412--1421.

Misra, D. (2019). Mish: A Self Regularized Non-Monotonic Activation Function. *arXiv preprint arXiv:1908.08681*.

Nair, V., & Hinton, G. E. (2010). Rectified Linear Units Improve Restricted Boltzmann Machines. *Proceedings of the 27th International Conference on Machine Learning*, 807--814.

Peng, B., Alcaide, E., Anthony, Q., et al. (2023). RWKV: Reinventing RNNs for the Transformer Era. *Findings of the Association for Computational Linguistics: EMNLP 2023*, 14048--14077.

Press, O., & Wolf, L. (2017). Using the Output Embedding to Improve Language Models. *Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics*, 157--163.

Ramachandran, P., Zoph, B., & Le, Q. V. (2017). Searching for Activation Functions. *arXiv preprint arXiv:1710.05941*.

Shazeer, N., Mirhoseini, A., Maziarz, K., et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *Proceedings of the 5th International Conference on Learning Representations*.

Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. *Proceedings of the 3rd International Conference on Learning Representations*.

Szegedy, C., Liu, W., Jia, Y., et al. (2015). Going Deeper with Convolutions. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 1--9.

Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *Proceedings of the 36th International Conference on Machine Learning*, 6105--6114.

Tay, Y., Dehghani, M., Bahri, D., & Metzler, D. (2022). Efficient Transformers: A Survey. *ACM Computing Surveys*, 55(6), 1--28.

Telgarsky, M. (2016). Benefits of Depth in Neural Networks. *Proceedings of the 29th Annual Conference on Learning Theory*, 1517--1539.

Touvron, H., Cord, M., Douze, M., et al. (2021). Training Data-Efficient Image Transformers & Distillation through Attention. *Proceedings of the 38th International Conference on Machine Learning*, 10347--10357.

Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems*, 30, 5998--6008.

Veit, A., Wilber, M. J., & Belongie, S. (2016). Residual Networks Behave Like Ensembles of Relatively Shallow Networks. *Advances in Neural Information Processing Systems*, 29, 550--558.

Xiong, R., Yang, Y., He, D., et al. (2020). On Layer Normalization in the Transformer Architecture. *Proceedings of the 37th International Conference on Machine Learning*, 10524--10533.
