# Chapter 8: Fine-Tuning and Adaptation

> *"The secret of getting ahead is getting started."* -- Mark Twain

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain why transfer learning and fine-tuning work from a representation learning perspective.
2. Calculate the memory requirements for full fine-tuning and compare them against parameter-efficient alternatives.
3. Derive the mathematical foundations of LoRA, QLoRA, DoRA, and IA3.
4. Implement mixed precision training with proper loss scaling strategies.
5. Apply gradient checkpointing to reduce activation memory by trading compute for memory.
6. Execute a complete fine-tuning pipeline using Hugging Face PEFT and TRL on a modern open-weight LLM.

---

## 8.1 Transfer Learning: Why Fine-Tuning Works

The story of modern NLP is, in large part, the story of transfer learning. A model pretrained on billions of tokens of text learns representations that capture syntax, semantics, world knowledge, and even rudimentary reasoning. Fine-tuning adapts these learned representations to a specific downstream task or domain with comparatively little data and compute.

### 8.1.1 Feature Reuse and Hierarchical Representations

Deep neural networks learn hierarchical features. In the context of language models, early layers capture low-level linguistic features --- tokenization artifacts, positional patterns, and basic syntactic structures. Middle layers encode richer semantic representations, capturing word sense, entity types, and relational knowledge. Later layers specialize toward the pretraining objective, encoding patterns most directly useful for next-token prediction (Howard & Ruder, 2018).

When we fine-tune, we leverage a powerful insight: **most of these features are task-agnostic**. The understanding of English grammar learned from pretraining is equally useful for sentiment classification, question answering, or code generation. Fine-tuning merely adjusts the upper layers (and to a lesser degree, the lower layers) to map these general representations onto the specific output space required by the downstream task.

Formally, consider a pretrained model as a function $f_\theta(x) = h \circ g(x)$, where $g: \mathcal{X} \to \mathcal{Z}$ maps inputs to a representation space and $h: \mathcal{Z} \to \mathcal{Y}$ maps representations to outputs. Transfer learning works when the representation $g$ learned on the source distribution $P_S(X, Y)$ is also useful for the target distribution $P_T(X, Y)$.

The theoretical justification rests on the assumption that the source and target distributions share common structure. Ben-David et al. (2010) formalized this through domain adaptation theory, showing that the target error is bounded by the source error plus a divergence term measuring how different the two distributions are. For language models pretrained on broad web corpora, this divergence is small for most downstream NLP tasks, because the pretraining data already contains examples of nearly every genre, topic, and writing style.

Probing studies provide direct evidence for the hierarchical nature of learned representations. Tenney et al. (2019) showed that BERT's representations recapitulate the classical NLP pipeline: POS tagging is best predicted from early layers, syntactic parsing from middle layers, and semantic role labeling from later layers. Similar patterns have been observed in decoder-only models like GPT-2 (Vig, 2019), where attention heads in early layers track syntactic dependencies and later layers capture more abstract semantic relationships.

### 8.1.2 The Intrinsic Dimensionality Hypothesis

Aghajanyan et al. (2021) made a remarkable observation: the fine-tuning process operates in a surprisingly low-dimensional subspace. Using the concept of intrinsic dimensionality --- the minimum number of parameters needed to achieve 90% of the full fine-tuning performance --- they found that common NLU tasks have intrinsic dimensionalities of only 100--1000, even for models with hundreds of millions of parameters.

This finding has two important implications. First, it explains why parameter-efficient fine-tuning methods work so well: if the effective dimensionality of fine-tuning is low, we do not need to update all parameters. Second, it suggests that pretraining places the model in a region of parameter space from which many downstream tasks are reachable through small perturbations, a phenomenon they called "pre-training implicitly minimizes intrinsic dimensionality."

### 8.1.3 Domain Shift and When to Fine-Tune vs. Feature Extract

The effectiveness of fine-tuning depends on the degree of **domain shift** between the pretraining data and the target task. We can characterize two axes:

1. **Data similarity**: How similar is the target domain's text to the pretraining corpus?
2. **Task similarity**: How close is the target task to the pretraining objective?

This yields a practical decision matrix:

| | Similar Task | Different Task |
|---|---|---|
| **Similar Data** | Feature extraction or light fine-tuning | Fine-tune upper layers |
| **Different Data** | Fine-tune with care (risk of catastrophic forgetting) | Full fine-tuning or continued pretraining first |

**Feature extraction** --- freezing the pretrained model and training only a new head --- works well when the domains are similar and data is scarce. **Fine-tuning** the entire model (or a substantial portion) becomes necessary when there is significant distributional shift, as the pretrained representations must be reshaped to accommodate new patterns.

A critical failure mode is **catastrophic forgetting**, where fine-tuning on a narrow task erases the general capabilities learned during pretraining. When a model is fine-tuned on sentiment classification, for example, its ability to perform named entity recognition or question answering may degrade significantly. The severity of forgetting depends on several factors: the size of the fine-tuning dataset (larger datasets cause more forgetting), the learning rate (higher rates overwrite more aggressively), and the dissimilarity between the fine-tuning task and the pretraining objective.

Techniques to mitigate catastrophic forgetting include:

- **Low learning rates**: Limiting the magnitude of weight updates preserves more of the pretrained representations.
- **Freezing early layers**: Since early layers capture general linguistic features, keeping them fixed preserves the foundation while allowing upper layers to specialize.
- **Regularization**: Elastic Weight Consolidation (EWC) (Kirkpatrick et al., 2017) adds a penalty for moving important parameters far from their pretrained values, where importance is measured by the Fisher information matrix.
- **Parameter-efficient methods**: Methods like LoRA that limit the number of modified parameters inherently reduce forgetting because most weights remain unchanged.
- **Data replay**: Mixing a small proportion of pretraining-style data into the fine-tuning batches helps maintain general capabilities.

### 8.1.4 Layer-wise Learning Rate Strategies

Not all layers should be updated equally during fine-tuning. Howard & Ruder (2018) introduced **discriminative fine-tuning**, which assigns different learning rates to different layers:

$$\eta^l = \eta^L \cdot \xi^{L-l}$$

where $\eta^L$ is the learning rate for the final layer, $l$ is the layer index, $L$ is the total number of layers, and $\xi < 1$ is a decay factor (typically 0.95). This assigns lower learning rates to earlier layers, reflecting the intuition that general features should be modified less than task-specific ones.

They also introduced **slanted triangular learning rates**, which rapidly increase the learning rate at the beginning of training (to quickly adapt the upper layers) and then gradually decay it (to fine-tune the lower layers). This schedule proved more effective than constant or simple decaying schedules for text classification tasks.

These layer-wise strategies are less commonly used with modern parameter-efficient methods (which achieve a similar effect by restricting which parameters are modified), but they remain valuable for full fine-tuning scenarios.

---

## 8.2 Full Fine-Tuning at Scale

Despite the popularity of parameter-efficient methods, full fine-tuning remains the gold standard when compute and memory are available. All parameters are updated via gradient descent, giving the optimizer maximum flexibility to adapt the model.

### 8.2.1 When Full Fine-Tuning Beats PEFT

Full fine-tuning consistently outperforms parameter-efficient methods when:

- **The target domain differs substantially** from the pretraining distribution (e.g., adapting an English model to a low-resource language).
- **Sufficient high-quality data** is available (tens of thousands to millions of examples).
- **Maximum performance** is required and the computational budget permits it.
- **The model is relatively small** (under 3B parameters), where PEFT savings are less critical.

Empirically, Hu et al. (2021) showed that LoRA can match full fine-tuning on many benchmarks, but subsequent work has identified tasks --- particularly those requiring significant distribution shift --- where full fine-tuning retains an edge (Biderman et al., 2024).

### 8.2.2 Optimizer State Management

The memory cost of fine-tuning is dominated not by the model parameters themselves, but by the **optimizer states**. Consider AdamW, the standard optimizer for transformer training. For each parameter $\theta_i$, AdamW maintains:

- **First moment estimate** $m_i$ (same shape as $\theta_i$)
- **Second moment estimate** $v_i$ (same shape as $\theta_i$)

This means AdamW requires **3x the model parameter memory** (parameters + two optimizer states) when using full precision, or **2x additional** beyond the parameter storage.

### 8.2.3 Memory Requirements Calculation

For a model with $N$ parameters, the total GPU memory required for full fine-tuning in mixed precision is approximately:

$$M_{\text{total}} = M_{\text{params}} + M_{\text{grads}} + M_{\text{optim}} + M_{\text{activations}}$$

Breaking this down for mixed precision (FP16/BF16 forward, FP32 optimizer):

| Component | Bytes per Parameter | For 7B Model |
|---|---|---|
| Model parameters (FP16) | 2 | 14 GB |
| Gradients (FP16) | 2 | 14 GB |
| Optimizer states (FP32): master weights | 4 | 28 GB |
| Optimizer states (FP32): $m$ and $v$ | 8 | 56 GB |
| **Subtotal (excluding activations)** | **16** | **~112 GB** |

Activation memory scales with batch size, sequence length, and model depth:

$$M_{\text{activations}} \approx 2 \cdot L \cdot s \cdot b \cdot h \cdot d$$

where $L$ is the number of layers, $s$ is the sequence length, $b$ is the batch size, $h$ is the hidden dimension, and $d$ is a constant depending on the architecture (typically 10--16 bytes per element for standard transformers). For a 7B model with sequence length 2048 and batch size 1, activation memory is typically 10--20 GB.

The implication is sobering: **full fine-tuning a 7B model requires approximately 120--130 GB of GPU memory**, which exceeds the capacity of a single A100 80GB. This motivates both distributed training (Part IV) and parameter-efficient methods.

### 8.2.4 Practical Guidelines for Full Fine-Tuning

1. **Learning rate**: Use $1\mathrm{e}{-5}$ to $5\mathrm{e}{-5}$ --- much lower than pretraining rates. The pretrained weights are already in a good region of the loss landscape.
2. **Warmup**: Use 3--10% of total steps. This stabilizes early training when gradients may be noisy.
3. **Weight decay**: Maintain the same weight decay used during pretraining (typically 0.1).
4. **Epochs**: For instruction tuning, 1--3 epochs is typical. More epochs risk overfitting and catastrophic forgetting.
5. **Gradient clipping**: Clip to norm 1.0 to prevent gradient explosions from outlier batches.

---

## 8.3 LoRA: Low-Rank Adaptation

Low-Rank Adaptation (LoRA) introduced by Hu et al. (2021) is arguably the most influential parameter-efficient fine-tuning method. Its elegance lies in a simple observation: the weight updates during fine-tuning have low intrinsic rank.

### 8.3.1 Mathematical Derivation

Consider a pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$. During fine-tuning, the weight update $\Delta W$ also resides in $\mathbb{R}^{d \times k}$. LoRA hypothesizes that $\Delta W$ has a low intrinsic rank $r \ll \min(d, k)$.

We decompose $\Delta W$ as:

$$\Delta W = BA$$

where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$. The modified forward pass becomes:

$$h = W_0 x + \Delta W x = W_0 x + BAx$$

The number of trainable parameters drops from $d \times k$ to $r \times (d + k)$. For a typical attention weight where $d = k = 4096$ and $r = 16$, this is a reduction from 16.8M to 131K parameters --- a **128x reduction**.

**Initialization.** $A$ is initialized from a Gaussian distribution $\mathcal{N}(0, \sigma^2)$ and $B$ is initialized to zero. This ensures that $\Delta W = BA = 0$ at the start of training, so the model begins from the pretrained weights exactly.

**Scaling.** An additional scaling factor $\alpha / r$ is applied:

$$h = W_0 x + \frac{\alpha}{r} BAx$$

The hyperparameter $\alpha$ (typically set to 16 or 32) controls the magnitude of the LoRA update relative to the pretrained weights. When $\alpha = r$, the scaling factor is 1. When $\alpha > r$, the LoRA updates are amplified. The ratio $\alpha / r$ acts as a learning rate multiplier for the LoRA parameters.

**Why does the low-rank hypothesis hold?** Consider what happens during fine-tuning. The pretrained weights $W_0$ already encode general-purpose language understanding. The update $\Delta W$ needs to encode only the task-specific adjustments --- for example, shifting from next-token prediction to instruction following, or from general English to medical terminology. These adjustments are specific and structured, requiring far fewer degrees of freedom than the original representation. The singular value decomposition of empirically observed $\Delta W$ matrices confirms this: the spectrum of singular values decays rapidly, with most of the energy concentrated in the top few singular values.

### 8.3.2 Rank Selection

The rank $r$ controls the expressiveness of the adaptation:

- **$r = 1$--$4$**: Extremely parameter-efficient. Suitable for simple tasks or when data is scarce. At $r = 1$, the update is a rank-1 matrix $\mathbf{b}\mathbf{a}^\top$, which can only adjust one "direction" per weight matrix.
- **$r = 8$--$16$**: The most common choice. Provides a good balance between expressiveness and efficiency. Hu et al. (2021) found $r = 8$ sufficient for most NLU tasks.
- **$r = 32$--$64$**: Approaches full fine-tuning expressiveness. Useful for complex tasks, significant domain shift, or tasks requiring the model to learn genuinely new patterns (e.g., a new language).
- **$r = 256+$**: Diminishing returns. At this point, full fine-tuning may be more straightforward and gives the optimizer more flexibility.

Hu et al. (2021) found that performance plateaus surprisingly quickly as $r$ increases, supporting the low-rank hypothesis. For most practical applications, $r = 16$ is a robust default.

**Adaptive rank selection.** Recent work on AdaLoRA (Zhang et al., 2023) introduced adaptive rank allocation, where the rank is learned during training. Different weight matrices receive different ranks based on their importance for the downstream task. AdaLoRA parameterizes $\Delta W$ using its singular value decomposition $\Delta W = P \Lambda Q$ and prunes singular values with the smallest importance scores, effectively learning the optimal rank per layer.

### 8.3.3 Target Modules

Not all weight matrices benefit equally from LoRA adaptation. The standard choices are:

- **Query and Value projections** ($W_Q$, $W_V$): The original LoRA paper found these most effective.
- **All attention projections** ($W_Q$, $W_K$, $W_V$, $W_O$): A common practical choice that improves over Q/V-only.
- **All linear layers** (attention + MLP): Maximum expressiveness, commonly used in practice with modern implementations.

The choice of target modules interacts with the rank: applying LoRA to more modules with a smaller rank can outperform applying it to fewer modules with a larger rank, for the same total parameter count.

### 8.3.4 Merge and Unmerge

A key advantage of LoRA is that the adapted weights can be **merged** back into the base model:

$$W_{\text{merged}} = W_0 + \frac{\alpha}{r} BA$$

After merging, the model has the same architecture and inference cost as the original --- no additional latency. This also enables **adapter switching**: multiple LoRA adapters can be swapped in and out by merging/unmerging, allowing a single base model to serve multiple tasks.

```python
from peft import PeftModel

# Load base model + adapter
model = PeftModel.from_pretrained(base_model, "path/to/adapter")

# Merge LoRA weights into base model
model = model.merge_and_unload()

# Now model is a standard model with no adapter overhead
model.save_pretrained("merged_model")
```

**Multi-adapter serving.** In production deployments, a single base model can serve hundreds of different tasks by loading the appropriate LoRA adapter per request. Libraries like LoRAX and S-LoRA enable efficient multi-adapter serving by batching requests across different adapters and sharing the base model weights in GPU memory. The adapter weights themselves are small (typically 10--50 MB), making it feasible to keep hundreds in memory simultaneously.

**Adapter composition.** Multiple LoRA adapters can be composed in several ways:

- **Linear combination**: $W = W_0 + \alpha_1 B_1 A_1 + \alpha_2 B_2 A_2$, where each adapter contributes to the final weights. The mixing coefficients $\alpha_i$ can be tuned or learned.
- **Sequential application**: Apply adapters one after another, fine-tuning on top of a previously adapted model.
- **Task arithmetic**: Ilharco et al. (2023) showed that arithmetic operations on task vectors (the difference between fine-tuned and pretrained weights) can combine capabilities, negate undesirable behaviors, or interpolate between tasks.

### 8.3.5 LoRA Implementation from Scratch

To solidify understanding, let us implement a LoRA linear layer from scratch:

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """Linear layer with Low-Rank Adaptation."""

    def __init__(self, in_features, out_features, r=16, alpha=32,
                 dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Original (frozen) weight
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        self.bias = None

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Dropout on LoRA path
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B is already initialized to zeros

    def forward(self, x):
        # Original path (frozen weights)
        result = nn.functional.linear(x, self.weight, self.bias)

        # LoRA path
        lora_input = self.lora_dropout(x)
        lora_output = lora_input @ self.lora_A.T @ self.lora_B.T
        result = result + lora_output * self.scaling

        return result

    def merge(self):
        """Merge LoRA weights into the base weight matrix."""
        self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
        # Optionally reset LoRA matrices
        self.lora_A.data.zero_()
        self.lora_B.data.zero_()
```

---

## 8.4 QLoRA: Quantized Low-Rank Adaptation

QLoRA (Dettmers et al., 2023) combines LoRA with aggressive quantization of the base model, enabling fine-tuning of 65B parameter models on a single 48GB GPU. It introduces three technical innovations: 4-bit NormalFloat quantization, double quantization, and paged optimizers.

### 8.4.1 NF4 Quantization: Why Normal Float?

Standard integer quantization maps weights uniformly across the representable range. However, pretrained neural network weights are not uniformly distributed --- they follow an approximately **normal distribution** centered at zero.

NormalFloat (NF4) exploits this observation. It defines 16 quantization levels (4 bits = $2^4$ values) that are **uniformly spaced in the quantile space** of the standard normal distribution. Formally, the quantization levels $q_i$ are defined as:

$$q_i = \Phi^{-1}\left(\frac{i}{2^b}\right), \quad i = 0, 1, \ldots, 2^b - 1$$

where $\Phi^{-1}$ is the inverse cumulative distribution function of the standard normal. This ensures that each quantization bin contains approximately the same number of weights, minimizing the expected quantization error.

Compared to standard INT4 quantization, NF4 reduces quantization error by approximately 50% for normally distributed weights.

To understand why this matters, consider a concrete example. With uniform INT4 quantization over the range $[-1, 1]$, the quantization levels are equally spaced: $\{-1, -0.867, -0.733, \ldots, 0.733, 0.867, 1\}$. Since most weights cluster near zero (the peak of the normal distribution), the bins near zero contain many weights while the bins near the extremes contain very few. This wastes representational capacity.

NF4 instead places more quantization levels where the weights are densest (near zero) and fewer where weights are sparse (near the tails). The resulting quantization levels for NF4 are approximately:

$$\{-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0\}$$

Note the asymmetric spacing: levels are densely packed near zero and sparse near the extremes.

**Block-wise quantization.** In practice, NF4 is applied block-wise: the weights are divided into blocks of 64 consecutive values, each block is normalized to have a maximum absolute value of 1 (using a per-block scale factor), and then each normalized weight is quantized to the nearest NF4 level. The per-block scale factor allows different parts of the weight matrix to use different ranges, improving accuracy.

### 8.4.2 Double Quantization

Each block of weights (typically 64 weights) requires a quantization constant (scale factor) stored in FP32 (4 bytes). For a 7B parameter model, this adds approximately $7 \times 10^9 / 64 \times 4 = 437.5$ MB of overhead.

**Double quantization** quantizes these quantization constants themselves to FP8, reducing the overhead to approximately 110 MB --- a 4x reduction. The second-level quantization constants are stored in FP32, but since there are far fewer of them, the overhead is negligible.

### 8.4.3 Paged Optimizers

During training, GPU memory usage spikes during gradient computation. QLoRA uses NVIDIA's unified memory feature to page optimizer states between GPU and CPU memory:

- During the forward and backward pass, only the currently needed optimizer states reside in GPU memory.
- States for other layers are paged to CPU memory.
- This prevents out-of-memory errors during gradient spikes.

The implementation leverages the `bitsandbytes` library's paged optimizers:

```python
import bitsandbytes as bnb

optimizer = bnb.optim.AdamW8bit(
    model.parameters(),
    lr=2e-4,
    is_paged=True  # Enable paged optimizer
)
```

### 8.4.4 Memory Savings

Let us compare the memory requirements for fine-tuning a 7B model:

| Method | Model | Gradients | Optimizer | Total (approx.) |
|---|---|---|---|---|
| Full FT (FP16) | 14 GB | 14 GB | 84 GB | ~130 GB |
| LoRA (FP16 base) | 14 GB | ~0.1 GB | ~0.6 GB | ~15 GB |
| QLoRA (NF4 base) | 3.5 GB | ~0.1 GB | ~0.6 GB | ~5 GB |

QLoRA achieves a **26x memory reduction** compared to full fine-tuning, with minimal loss in performance. Dettmers et al. (2023) demonstrated that QLoRA matches full 16-bit fine-tuning performance across a range of benchmarks.

---

## 8.5 DoRA: Weight-Decomposed Low-Rank Adaptation

DoRA (Liu et al., 2024) builds on LoRA by decomposing the weight matrix into **magnitude** and **direction** components, inspired by weight normalization (Salimans & Kingma, 2016).

### 8.5.1 Mathematical Formulation

In standard LoRA, the modified weight is:

$$W' = W_0 + BA$$

DoRA decomposes this differently. First, recall that any weight matrix can be expressed as:

$$W = m \cdot \frac{V}{\|V\|_c}$$

where $m \in \mathbb{R}^{1 \times k}$ is a magnitude vector (one scalar per output column), $V \in \mathbb{R}^{d \times k}$ is the directional component, and $\|V\|_c$ denotes the column-wise norm.

DoRA makes the magnitude $m$ a trainable parameter and applies LoRA to the directional component:

$$W' = m \cdot \frac{W_0 + BA}{\|W_0 + BA\|_c}$$

The intuition is that **magnitude and direction capture fundamentally different aspects of the weight transformation**, and decoupling them allows more efficient adaptation. The magnitude controls the scaling of each output dimension, while the direction controls the feature mixing.

### 8.5.2 Comparison to LoRA

Liu et al. (2024) showed that:

- DoRA consistently outperforms LoRA across commonsense reasoning, visual instruction tuning, and image-text understanding tasks.
- The improvement is most pronounced at low ranks ($r = 4$--$8$), where LoRA's expressiveness is most constrained.
- DoRA's learning pattern more closely resembles that of full fine-tuning, particularly in terms of the relative magnitudes of directional vs. magnitude updates.

The additional computational overhead of DoRA is minimal: one extra norm computation per adapted layer during the forward pass.

### 8.5.3 Why Does Decoupling Magnitude and Direction Help?

The intuition behind DoRA can be understood through the geometry of weight updates. During full fine-tuning, Liu et al. (2024) observed that the weight changes exhibit two distinct components:

1. **Magnitude changes**: The overall scale of each column changes, adjusting how strongly each input feature influences the output.
2. **Directional changes**: The orientation of each column in the weight space changes, adjusting the mixture of input features.

Standard LoRA couples these two components: the low-rank update $BA$ modifies both magnitude and direction simultaneously. This coupling means that to achieve a desired directional change, the rank must be high enough to also capture the correlated magnitude change (and vice versa). DoRA breaks this coupling, allowing each component to be optimized independently.

In practice, DoRA adds very few parameters beyond LoRA --- only the magnitude vector $m \in \mathbb{R}^k$, which has $k$ elements per adapted layer (compared to $(d + k) \times r$ for LoRA). For $d = k = 4096$ and $r = 16$, the magnitude vector adds 4096 parameters, a 3% increase over LoRA's 131,072 parameters per layer.

### 8.5.4 DoRA Implementation

```python
class DoRALinear(nn.Module):
    """Linear layer with Weight-Decomposed Low-Rank Adaptation."""

    def __init__(self, linear_layer, r=16, alpha=32):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        weight = linear_layer.weight.data  # (out_features, in_features)
        out_features, in_features = weight.shape

        # Decompose pretrained weight into magnitude and direction
        col_norms = weight.norm(dim=0, keepdim=True)  # (1, in_features)
        self.magnitude = nn.Parameter(col_norms.squeeze())  # trainable
        self.direction = nn.Parameter(weight / col_norms, requires_grad=False)  # frozen

        # LoRA matrices for the directional component
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

    def forward(self, x):
        # Updated directional component
        updated_direction = self.direction + (self.lora_B @ self.lora_A) * self.scaling
        # Normalize columns
        col_norms = updated_direction.norm(dim=0, keepdim=True)
        normalized_direction = updated_direction / col_norms
        # Apply magnitude
        weight = self.magnitude.unsqueeze(0) * normalized_direction
        return nn.functional.linear(x, weight)
```

---

## 8.6 IA3: Infused Adapter by Inhibiting and Amplifying Inner Activations

IA3 (Liu et al., 2022) takes a different approach to parameter efficiency. Rather than adding low-rank matrices, IA3 learns **scaling vectors** that rescale the keys, values, and feed-forward activations.

### 8.6.1 Formulation

For each target activation, IA3 introduces a learned vector $l \in \mathbb{R}^d$ that performs element-wise multiplication:

$$h_{\text{adapted}} = l \odot h_{\text{original}}$$

Specifically, IA3 rescales:
- Keys in attention: $k' = l_k \odot (W_K x)$
- Values in attention: $v' = l_v \odot (W_V x)$
- Feed-forward intermediate: $f' = l_f \odot (W_{\text{ff}} x)$

### 8.6.2 Parameter Efficiency

IA3 is remarkably parameter-efficient. For a transformer layer with hidden dimension $d$ and intermediate dimension $d_{\text{ff}}$:

- LoRA (rank 16, Q/V projections): $2 \times 16 \times (d + d) = 64d$ parameters per layer
- IA3: $2d + d_{\text{ff}}$ parameters per layer

For a model where $d_{\text{ff}} = 4d$ (common in transformers), IA3 uses $6d$ parameters per layer --- roughly **10x fewer than LoRA** at rank 16. However, this extreme efficiency comes at a cost: IA3 generally underperforms LoRA on tasks requiring significant adaptation, as it can only scale existing features, not create new ones.

---

## 8.7 Prefix Tuning, Prompt Tuning, and P-Tuning

These methods add **trainable tokens** to the input or intermediate representations, leaving the model weights entirely frozen.

### 8.7.1 Prefix Tuning

Prefix Tuning (Li & Liang, 2021) prepends trainable continuous vectors --- "virtual tokens" --- to the keys and values at every attention layer. For a prefix length $l$:

$$\text{Attention}(Q, [P_K; K], [P_V; V])$$

where $P_K, P_V \in \mathbb{R}^{l \times d}$ are the trainable prefix matrices. The prefixes are not constrained to the model's token embedding space, giving them more expressiveness.

To stabilize training, Li & Liang (2021) reparameterize the prefix through a small feed-forward network:

$$P = \text{MLP}(P_{\text{init}})$$

After training, the MLP is discarded and only the final prefix values $P$ are saved.

**Number of trainable parameters**: For prefix length $l$, $L$ layers, and hidden dimension $d$:

$$N_{\text{params}} = 2 \times l \times L \times d$$

(The factor of 2 accounts for separate key and value prefixes.)

### 8.7.2 Prompt Tuning

Prompt Tuning (Lester et al., 2021) is a simplified version that prepends trainable embeddings only at the input layer:

$$\tilde{X} = [P; X]$$

where $P \in \mathbb{R}^{l \times d}$ is a trainable soft prompt. This is far simpler than prefix tuning but surprisingly effective for large models. Lester et al. (2021) showed that as model size increases, prompt tuning closes the gap with full fine-tuning, nearly matching it at 10B+ parameters.

### 8.7.3 P-Tuning v2

P-tuning v2 (Liu et al., 2022) applies trainable continuous prompts at every layer of the transformer (like prefix tuning) but specifically targets the setting where the model is used for NLU tasks. It adds prefix tokens to each layer and demonstrated that prefix tuning can match full fine-tuning even for smaller models (330M--10B parameters) across a variety of NLU benchmarks, provided the prefix length and training recipe are properly tuned.

---

## 8.8 Mixed Precision Training

Mixed precision training uses lower-precision numerical formats (FP16 or BF16) for the forward and backward passes while maintaining FP32 master copies of weights for optimizer updates. This reduces memory usage and increases throughput by exploiting the hardware's tensor cores (Micikevicius et al., 2018).

### 8.8.1 FP16 vs. BF16 Numerical Properties

**FP16 (Half Precision)**:
- 1 sign bit, 5 exponent bits, 10 mantissa bits
- Range: $\pm 6.55 \times 10^4$
- Precision: ~3.3 decimal digits
- Risk: values outside the representable range cause overflow (inf) or underflow (0)

**BF16 (Brain Floating Point)**:
- 1 sign bit, 8 exponent bits, 7 mantissa bits
- Range: $\pm 3.4 \times 10^{38}$ (same as FP32!)
- Precision: ~2.4 decimal digits
- Advantage: same exponent range as FP32, so overflow/underflow is extremely rare

The key difference: **BF16 trades precision for range**. Since gradient values in deep learning span many orders of magnitude, BF16's wider range makes it far more robust. FP16's limited range (max ~65504) frequently causes overflow during training, necessitating loss scaling.

### 8.8.2 torch.autocast and GradScaler

PyTorch's `torch.autocast` context manager automatically selects the appropriate precision for each operation:

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# FP16 training requires GradScaler
scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    with autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(batch['input_ids'])
        loss = criterion(outputs, batch['labels'])

    # Scale loss to prevent gradient underflow in FP16
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
```

**Why GradScaler is needed for FP16 but not BF16**: FP16's limited dynamic range means that small gradient values (common in deep networks) underflow to zero. The `GradScaler` multiplies the loss by a large factor (e.g., 1024) before the backward pass, scaling all gradients up into the representable range. After the backward pass, gradients are unscaled before the optimizer step.

BF16, with its FP32-equivalent range, does not suffer from this problem:

```python
# BF16 training — no GradScaler needed
for batch in dataloader:
    optimizer.zero_grad()

    with autocast(device_type='cuda', dtype=torch.bfloat16):
        outputs = model(batch['input_ids'])
        loss = criterion(outputs, batch['labels'])

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

### 8.8.3 Loss Scaling Details

The `GradScaler` implements dynamic loss scaling:

1. Start with a large scale factor (e.g., $2^{16} = 65536$).
2. If no infs/NaNs are detected in the gradients, the optimizer step proceeds normally. The scale factor is increased periodically (e.g., multiplied by 2 every 2000 steps).
3. If infs/NaNs are detected, the optimizer step is skipped and the scale factor is halved.

This adaptive scheme finds the largest stable scale factor, maximizing the use of FP16's dynamic range.

**Why does loss scaling work?** Consider a gradient value of $10^{-8}$, which is well below FP16's smallest normal number ($\approx 6 \times 10^{-8}$) and would underflow to zero. If we multiply the loss by $2^{16} = 65536$ before the backward pass, all gradients are scaled up by the same factor, so this gradient becomes $10^{-8} \times 65536 \approx 6.5 \times 10^{-4}$, which is comfortably within FP16's range. After the backward pass, we divide all gradients by the same scale factor before the optimizer step, recovering the correct gradient values.

### 8.8.4 Which Operations Use Which Precision?

The `autocast` context manager does not simply cast all operations to FP16/BF16. It maintains a list of operations categorized by numerical sensitivity:

- **FP16/BF16-safe** (run in low precision): Matrix multiplications, convolutions --- these benefit most from tensor cores and are numerically stable.
- **FP32-required** (always run in FP32): Softmax, layer normalization, loss functions, reductions --- these involve operations (exp, log, sum of many values) that are sensitive to numerical range and precision.
- **Promotion** (cast inputs to the widest type): Operations with mixed-precision inputs are promoted to the wider type.

This selective casting is critical. A naive approach of running everything in FP16 would lead to training divergence because operations like softmax (which involves exponentiation) would overflow in FP16 for large logit values.

### 8.8.5 Practical Recommendations

For modern LLM fine-tuning, the practical recommendation is straightforward:

1. **Use BF16 if your hardware supports it** (A100, H100, RTX 3000/4000 series, Apple M-series). It is simpler (no GradScaler needed) and more robust.
2. **Fall back to FP16 with GradScaler** on older hardware (V100, RTX 2000 series) that lacks BF16 support.
3. **Never use pure FP32** for LLM fine-tuning --- it wastes memory and is 2--4x slower on modern GPUs.
4. **Monitor for NaN/Inf**: Even with BF16, numerical issues can arise from bad data or extremely high learning rates. Always log the loss and check for anomalies.

---

## 8.9 Gradient Checkpointing

Gradient checkpointing (also called activation checkpointing or rematerialization) trades compute for memory by **recomputing activations** during the backward pass instead of storing them (Chen et al., 2016).

### 8.9.1 The Memory Problem

During the forward pass, every intermediate activation must be saved for use in the backward pass. For a transformer with $L$ layers, the activation memory scales as:

$$M_{\text{activations}} = O(L \cdot s \cdot b \cdot d)$$

where $s$ is the sequence length, $b$ is the batch size, and $d$ is the hidden dimension. For large models with long sequences, this can exceed the model parameter memory.

### 8.9.2 Activation Recomputation

Gradient checkpointing divides the model into segments. During the forward pass, only the activations at segment boundaries (checkpoints) are saved. During the backward pass, when activations for a segment are needed, the forward pass for that segment is recomputed from its checkpoint.

**Memory savings formula**: If we checkpoint every $k$ layers out of $L$:

$$M_{\text{checkpointed}} = \frac{L}{k} \cdot M_{\text{per\_layer}} + k \cdot M_{\text{per\_layer}}$$

The optimal checkpoint interval is $k = \sqrt{L}$, giving:

$$M_{\text{optimal}} = 2\sqrt{L} \cdot M_{\text{per\_layer}}$$

This reduces activation memory from $O(L)$ to $O(\sqrt{L})$, at the cost of approximately **33% additional compute** (one extra forward pass per segment).

### 8.9.3 Implementation

In PyTorch:

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformerBlock(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        # Recompute this layer's forward pass during backward
        return checkpoint(self.layer, x, use_reentrant=False)
```

In Hugging Face Transformers:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model.gradient_checkpointing_enable()  # That's it!
```

This single line typically reduces activation memory by 50--70%, enabling larger batch sizes or longer sequences.

### 8.9.4 Selective Checkpointing

Not all layers benefit equally from checkpointing. Attention layers produce large intermediate tensors (the attention matrix is $O(s^2)$ per head), while MLP layers produce smaller intermediates. **Selective checkpointing** applies recomputation only to the most memory-intensive operations:

```python
# Selective checkpointing: only checkpoint attention, not MLP
class SelectiveCheckpointBlock(nn.Module):
    def __init__(self, attention, mlp, norm1, norm2):
        super().__init__()
        self.attention = attention
        self.mlp = mlp
        self.norm1 = norm1
        self.norm2 = norm2

    def forward(self, x):
        # Checkpoint the attention computation (large activations)
        attn_out = checkpoint(
            lambda h: self.attention(self.norm1(h)),
            x, use_reentrant=False
        )
        x = x + attn_out
        # Do NOT checkpoint the MLP (smaller activations, faster to just store)
        x = x + self.mlp(self.norm2(x))
        return x
```

This hybrid approach achieves most of the memory savings (since attention dominates activation memory at long sequences) with less computational overhead than full checkpointing.

### 8.9.5 Memory Savings in Practice

To illustrate the practical impact, consider a LLaMA-7B model with 32 layers, hidden dimension 4096, 32 attention heads, sequence length 4096, and batch size 1:

| Configuration | Activation Memory | Compute Overhead |
|---|---|---|
| No checkpointing | ~52 GB | 0% |
| Full checkpointing (every layer) | ~1.6 GB | +33% |
| Checkpointing every 2 layers | ~3.2 GB | +17% |
| Selective (attention only) | ~8 GB | +15% |

The choice depends on the constraint: if memory is the bottleneck (e.g., trying to fit on a smaller GPU), full checkpointing is warranted despite the compute overhead. If training speed is the priority, selective checkpointing provides a better tradeoff.

---

## 8.10 Continued Pretraining

Sometimes, neither feature extraction nor fine-tuning is sufficient. When the target domain is very different from the pretraining data (e.g., biomedical literature, legal contracts, source code in a niche language), **continued pretraining** --- also called domain-adaptive pretraining --- can bridge the gap.

### 8.10.1 When to Continue Pretraining vs. Fine-Tune

| Signal | Continued Pretraining | Direct Fine-Tuning |
|---|---|---|
| Domain vocabulary differs significantly | Yes | No |
| Model perplexity on target data is high | Yes | No |
| Large unlabeled domain corpus available | Yes | N/A |
| Task-specific labeled data is abundant | Not necessarily | Yes |
| Need domain-specific world knowledge | Yes | Possibly |

### 8.10.2 Domain Adaptation Examples

**Medical**: Models like BioMistral and PMC-LLaMA continue pretraining on PubMed abstracts, clinical notes, and medical textbooks. The resulting models show dramatically improved performance on medical QA benchmarks (USMLE-style questions), as they acquire specialized vocabulary (drug names, anatomical terms) and domain knowledge (treatment protocols, diagnostic reasoning).

**Legal**: Legal text has distinctive syntax (long subordinate clauses, precise terminology, cross-references) that general-purpose models handle poorly. Continued pretraining on legal corpora --- court decisions, statutes, contracts --- adapts the model to these patterns.

**Code**: Models like CodeLlama (Roziere et al., 2023) begin from the general-purpose Llama 2 and continue pretraining on 500B tokens of code. The key insight is that code pretraining does not destroy the model's natural language capabilities --- it extends them.

### 8.10.3 The Continued Pretraining Recipe

A successful continued pretraining run requires careful attention to several hyperparameters and design choices:

**Learning rate.** Use 50--100x lower than the original pretraining learning rate. A typical choice is $1\mathrm{e}{-5}$ to $5\mathrm{e}{-5}$. Use a cosine decay schedule with a brief warmup (1--5% of total steps). Starting from a higher rate risks destabilizing the pretrained representations; starting from a lower rate wastes compute on unnecessarily small updates.

**Data mixing.** Include some general-domain data (5--20% of each batch) to prevent catastrophic forgetting. This "replay" of diverse data maintains the model's general language understanding while the domain-specific data teaches new knowledge and patterns. The LLaMA 2 paper (Touvron et al., 2023) and CodeLlama (Roziere et al., 2023) both used data mixing during continued pretraining.

**Token budget.** Even 10--50B tokens of domain-specific data can yield significant improvements. The relationship between performance improvement and token count follows diminishing returns: the first billion tokens provide the largest gain, with subsequent data yielding progressively smaller improvements. A useful heuristic is to continue pretraining until the domain-specific validation loss plateaus.

**Evaluation.** Monitor both domain-specific benchmarks and general capability benchmarks to detect forgetting. A successful continued pretraining run shows improving domain metrics with minimal degradation on general benchmarks. If general capabilities degrade significantly, increase the proportion of general-domain data in the mix.

**Tokenizer considerations.** If the target domain has specialized vocabulary (e.g., chemical formulas, legal citation formats, programming language syntax), the existing tokenizer may fragment these tokens poorly. In some cases, extending the tokenizer vocabulary with domain-specific tokens and resizing the embedding layer can improve efficiency. However, this approach requires careful initialization of new embeddings and can complicate model merging and sharing.

### 8.10.4 The Fine-Tuning Pipeline: Continued Pretraining, Then Instruction Tuning

A common and effective pipeline for domain-specific LLMs combines continued pretraining with instruction tuning:

1. **Continued pretraining** on a large domain corpus (unlabeled), adapting the model's world knowledge and language patterns to the target domain.
2. **Instruction tuning** (SFT) on a smaller dataset of domain-specific instruction-response pairs, teaching the model to respond helpfully in the domain.
3. **Preference optimization** (DPO or RLHF) on domain-specific preference data, refining the model's outputs.

Each stage builds on the previous: continued pretraining provides the domain knowledge, SFT teaches the model to use that knowledge in a helpful way, and preference optimization polishes the outputs. Skipping the continued pretraining step and going directly to SFT works for domains close to the pretraining distribution but yields suboptimal results for highly specialized domains.

---

## 8.11 Complete Walkthrough: Fine-Tuning with QLoRA

Let us now put everything together with a complete, runnable fine-tuning pipeline. We will fine-tune a Llama-style model on an instruction-following dataset using QLoRA, with the Hugging Face PEFT and TRL libraries.

### 8.11.1 Environment Setup

```python
# Install dependencies
# pip install transformers peft trl bitsandbytes datasets accelerate torch

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
```

### 8.11.2 Quantization Configuration

```python
# Configure 4-bit quantization (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in BF16
    bnb_4bit_use_double_quant=True,      # Double quantization
)

# Load model in 4-bit
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```

### 8.11.3 LoRA Configuration

```python
# Prepare model for QLoRA training
model = prepare_model_for_kbit_training(model)

# Define LoRA configuration
lora_config = LoraConfig(
    r=16,                          # Rank
    lora_alpha=32,                 # Alpha scaling
    target_modules=[               # Which modules to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()
# Output: trainable params: 13,631,488 || all params: 6,751,350,784 || 0.20%
```

### 8.11.4 Dataset Preparation

```python
# Load an instruction-following dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train")

# Define the formatting function for chat-style training
def format_instruction(sample):
    if sample["input"]:
        text = f"""### Instruction:
{sample["instruction"]}

### Input:
{sample["input"]}

### Response:
{sample["output"]}"""
    else:
        text = f"""### Instruction:
{sample["instruction"]}

### Response:
{sample["output"]}"""
    return text

# Preview a formatted example
print(format_instruction(dataset[0]))
```

### 8.11.5 Training Configuration

```python
# Define training arguments
training_args = TrainingArguments(
    output_dir="./llama2-qlora-alpaca",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,       # Effective batch size = 16
    gradient_checkpointing=True,         # Save memory
    optim="paged_adamw_8bit",            # QLoRA paged optimizer
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    learning_rate=2e-4,
    bf16=True,                           # BF16 training
    max_grad_norm=1.0,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="wandb",                   # Optional: logging
)

# Create the SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args,
    formatting_func=format_instruction,
    max_seq_length=2048,
    packing=True,                        # Pack multiple examples per sequence
)
```

### 8.11.6 Training and Saving

```python
# Train!
trainer.train()

# Save the adapter weights (not the full model)
trainer.model.save_pretrained("./llama2-qlora-alpaca/final_adapter")

# To merge and save the full model:
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    "./llama2-qlora-alpaca/final_adapter",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./llama2-qlora-alpaca/merged_model")
tokenizer.save_pretrained("./llama2-qlora-alpaca/merged_model")
```

### 8.11.7 Inference

```python
# Load the merged model for inference
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="./llama2-qlora-alpaca/merged_model",
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

prompt = """### Instruction:
Explain the concept of transfer learning in machine learning.

### Response:
"""

output = generator(
    prompt,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)
print(output[0]["generated_text"])
```

### 8.11.8 Memory Profile

On a single A100 40GB GPU, this QLoRA configuration uses approximately:

- Model (NF4): ~3.5 GB
- LoRA parameters: ~0.05 GB
- Optimizer states (8-bit, paged): ~0.1 GB
- Activations (with gradient checkpointing): ~4--8 GB
- **Total: ~8--12 GB**

This fits comfortably on consumer GPUs like the RTX 4090 (24 GB), RTX 3090 (24 GB), or even the RTX 4080 (16 GB) with reduced batch size.

---

## 8.12 Summary

This chapter covered the spectrum of fine-tuning approaches, from full-parameter updates to highly efficient methods requiring less than 1% of the original parameters. The key takeaway is that **there is no single best method** --- the choice depends on the available compute, the degree of domain shift, and the performance requirements.

| Method | Trainable Params (7B model) | Memory | Performance |
|---|---|---|---|
| Full Fine-Tuning | 7B (100%) | ~130 GB | Best (upper bound) |
| LoRA (r=16, all linear) | ~14M (0.2%) | ~15 GB | Near full FT |
| QLoRA (r=16, all linear) | ~14M (0.2%) | ~5 GB | Near full FT |
| DoRA (r=16) | ~14M + magnitude | ~16 GB | Slightly > LoRA |
| IA3 | ~0.5M (0.007%) | ~14.5 GB | Good for simple tasks |
| Prefix Tuning (l=20) | ~2.6M (0.04%) | ~14.5 GB | Good for NLU |
| Prompt Tuning (l=20) | ~82K (0.001%) | ~14 GB | Best at scale |

The practical recommendation for most practitioners: **start with QLoRA** (r=16, target all linear layers, alpha=32) using the walkthrough in Section 8.11. This provides an excellent performance-to-cost ratio and works on consumer hardware.

---

## Exercises

1. **Memory Calculation**: Calculate the exact GPU memory required to fine-tune a 13B parameter model using (a) full fine-tuning with AdamW in mixed precision, (b) LoRA with rank 32 targeting all attention projections, and (c) QLoRA with the same LoRA configuration. Assume sequence length 2048 and batch size 1.

2. **LoRA Rank Ablation**: Fine-tune a small language model (e.g., GPT-2 small) on a text classification task using LoRA with ranks $r \in \{1, 4, 8, 16, 32, 64, 128\}$. Plot validation accuracy vs. rank and total trainable parameters vs. rank. At what rank does performance saturate?

3. **DoRA Implementation**: Implement DoRA from scratch in PyTorch. Given a linear layer `nn.Linear(in_features, out_features)`, write a `DoRALinear` class that decomposes the weight into magnitude and direction, applies LoRA to the directional component, and computes the forward pass. Verify that your implementation produces the same output as the pretrained layer when the LoRA matrices are zero-initialized.

4. **Mixed Precision Comparison**: Train the same model on the same data using (a) FP32, (b) FP16 with GradScaler, and (c) BF16 without GradScaler. Compare training loss curves, final performance, training speed, and memory usage. Are there cases where FP16 training diverges?

5. **Gradient Checkpointing Analysis**: Measure the actual memory savings and training speed impact of gradient checkpointing on a transformer model. Vary the checkpointing strategy (every layer, every 2 layers, every 4 layers) and plot memory vs. compute tradeoff.

6. **Continued Pretraining Experiment**: Take a general-purpose model and continue pretraining it on a specialized domain corpus (e.g., scientific papers from ArXiv). Evaluate the model on domain-specific and general benchmarks before and after continued pretraining. How much general capability is lost?

---

## References

- Biderman, S., et al. (2024). LoRA Learns Less and Forgets Less. *arXiv preprint arXiv:2405.09673*.
- Chen, T., Xu, B., Zhang, C., & Guestrin, C. (2016). Training Deep Nets with Sublinear Memory Cost. *arXiv preprint arXiv:1604.06174*.
- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized Language Models. *Advances in Neural Information Processing Systems, 36*.
- Howard, J., & Ruder, S. (2018). Universal Language Model Fine-tuning for Text Classification. *Proceedings of ACL 2018*, 328--339.
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv preprint arXiv:2106.09685*.
- Lester, B., Al-Rfou, R., & Constant, N. (2021). The Power of Scale for Parameter-Efficient Prompt Tuning. *Proceedings of EMNLP 2021*, 3045--3059.
- Li, X. L., & Liang, P. (2021). Prefix-Tuning: Optimizing Continuous Prompts for Generation. *Proceedings of ACL 2021*, 4582--4597.
- Liu, H., Tam, D., Muqeeth, M., Mohta, J., Huang, T., Bansal, M., & Raffel, C. (2022). Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning. *Advances in Neural Information Processing Systems, 35*.
- Liu, S., Wang, C., Yin, H., Molchanov, P., Wang, Y. C., Cheng, K., & Chen, M. (2024). DoRA: Weight-Decomposed Low-Rank Adaptation. *arXiv preprint arXiv:2402.09353*.
- Liu, X., Ji, K., Fu, Y., Tam, W. L., Du, Z., Yang, Z., & Tang, J. (2022). P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks. *Proceedings of ACL 2022*, 6589--6601.
- Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., Ginsburg, B., Houston, M., Kuchaiev, O., Venkatesh, G., & Wu, H. (2018). Mixed Precision Training. *Proceedings of ICLR 2018*.
- Roziere, B., et al. (2023). Code Llama: Open Foundation Models for Code. *arXiv preprint arXiv:2308.12950*.
- Salimans, T., & Kingma, D. P. (2016). Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks. *Advances in Neural Information Processing Systems, 29*.
