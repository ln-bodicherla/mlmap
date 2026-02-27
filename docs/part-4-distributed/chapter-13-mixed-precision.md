# Chapter 13: Mixed Precision and Memory Optimization

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain the bit-level structure of FP32, FP16, BF16, INT8, INT4, and NF4 number formats and their implications for deep learning.
2. Implement mixed-precision training with `torch.autocast` and `GradScaler`, and explain why BF16 eliminates the need for loss scaling.
3. Describe the algorithmic innovations in Flash Attention V1, V2, and V3 and their impact on memory and speed.
4. Compare post-training quantization methods (GPTQ, AWQ, GGUF) and select the appropriate method for a given deployment scenario.
5. Explain speculative decoding and prove its mathematical guarantee of lossless generation.
6. Describe how PagedAttention manages KV cache memory and why continuous batching improves serving throughput.
7. Apply KV cache optimization techniques including prefix caching, multi-query attention, and grouped-query attention.

---

## 13.1 Number Formats — FP32, FP16, BF16, INT8, INT4

The choice of numerical precision is one of the most impactful decisions in deep learning system design. Every halving of precision roughly doubles the throughput and halves the memory — but only if the reduced precision is sufficient to represent the quantities involved.

### IEEE 754 Floating-Point Representation

A floating-point number is represented as:

$$(-1)^s \times 2^{e - \text{bias}} \times (1 + m)$$

where $s$ is the sign bit, $e$ is the stored exponent, bias is a format-specific constant, and $m$ is the fractional mantissa.

### FP32 (Single Precision)

```
| 1 bit sign | 8 bits exponent | 23 bits mantissa |
|     s      |    eeeeeeee     | mmmmmmmmmmmmmmmmmmmmmmm |
```

- **Exponent bias:** 127
- **Dynamic range:** $\pm 1.18 \times 10^{-38}$ to $\pm 3.40 \times 10^{38}$
- **Precision:** ~7.2 decimal digits
- **Size:** 4 bytes

FP32 is the "safe default" — sufficient for virtually all training scenarios but wasteful of memory and compute.

### Special Values in IEEE 754

All floating-point formats share certain special representations:

- **Zero:** exponent = 0, mantissa = 0 (positive and negative zero exist)
- **Subnormal numbers:** exponent = 0, mantissa != 0. These provide gradual underflow — smaller numbers than the minimum normal value, at reduced precision. The implicit leading bit is 0 instead of 1: $(-1)^s \times 2^{1-\text{bias}} \times (0 + m)$
- **Infinity:** exponent = all 1s, mantissa = 0
- **NaN (Not a Number):** exponent = all 1s, mantissa != 0

Understanding subnormals is particularly important for deep learning: they provide a gradual transition to zero rather than an abrupt cliff, which helps prevent complete loss of gradient information in reduced-precision formats.

### FP16 (Half Precision)

```
| 1 bit sign | 5 bits exponent | 10 bits mantissa |
|     s      |    eeeee        | mmmmmmmmmm       |
```

- **Exponent bias:** 15
- **Dynamic range:** $\pm 6.10 \times 10^{-5}$ to $\pm 65504$
- **Precision:** ~3.3 decimal digits
- **Size:** 2 bytes

The critical limitation of FP16 is its **narrow dynamic range**. The maximum value is only 65504 — loss values or gradient magnitudes that exceed this will overflow to infinity. More importantly, the minimum representable positive subnormal value is $\approx 5.96 \times 10^{-8}$; gradients smaller than this will **underflow to zero**, which is a common problem during training, particularly in later layers of deep networks.

### BF16 (Brain Floating Point)

```
| 1 bit sign | 8 bits exponent | 7 bits mantissa |
|     s      |    eeeeeeee     | mmmmmmm         |
```

- **Exponent bias:** 127 (same as FP32)
- **Dynamic range:** Same as FP32 ($\pm 1.18 \times 10^{-38}$ to $\pm 3.40 \times 10^{38}$)
- **Precision:** ~2.4 decimal digits
- **Size:** 2 bytes

BF16 was designed by Google Brain specifically for deep learning. By keeping the same 8-bit exponent as FP32, it maintains the full dynamic range, eliminating the overflow and underflow problems of FP16. The trade-off is reduced precision (7 vs. 10 mantissa bits), which is acceptable for neural network training because:

1. Stochastic gradient descent is inherently noisy, so high precision in individual operations is unnecessary.
2. Weight updates are performed in FP32 (the master copy), so precision loss does not accumulate.

### INT8 (8-bit Integer)

```
| 8 bits |
| iiiiiiii |
```

- **Range (signed):** -128 to 127
- **Range (unsigned):** 0 to 255
- **No exponent:** requires a separate scale factor

INT8 quantization maps floating-point values to integers using a scale and zero-point:

$$x_{\text{int}} = \text{round}\left(\frac{x_{\text{float}}}{\text{scale}}\right) + \text{zero\_point}$$

INT8 is primarily used for **inference** quantization, achieving 2x memory reduction with minimal accuracy loss for most models.

### INT4 and NF4 (4-bit Formats)

**INT4** provides only 16 distinct values (-8 to 7 for signed), which is extremely coarse. Naive INT4 quantization causes significant accuracy degradation.

**NF4 (Normal Float 4-bit)** was introduced by Dettmers et al. (2023) for QLoRA. NF4 assigns 16 quantization levels that are optimal for normally distributed weights:

$$\text{NF4 levels} = \{-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0, \\
 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0\}$$

These levels are chosen such that each level represents an equal probability mass under a standard normal distribution. Since pretrained neural network weights are approximately normally distributed, NF4 minimizes quantization error.

### Dynamic Range Comparison

| Format | Min Positive (Normal) | Max Value | Decimal Digits | Bytes |
|---|---|---|---|---|
| FP32 | $1.18 \times 10^{-38}$ | $3.40 \times 10^{38}$ | 7.2 | 4 |
| FP16 | $6.10 \times 10^{-5}$ | $6.55 \times 10^{4}$ | 3.3 | 2 |
| BF16 | $1.18 \times 10^{-38}$ | $3.40 \times 10^{38}$ | 2.4 | 2 |
| INT8 | 1 (integer) | 127 | N/A | 1 |
| INT4 | 1 (integer) | 7 | N/A | 0.5 |

---

## 13.2 Mixed Precision Training in Practice

Mixed precision training (Micikevicius et al., 2018) uses lower precision (FP16 or BF16) for the forward and backward passes to reduce memory and increase throughput, while maintaining FP32 "master weights" for the optimizer update step to preserve training accuracy.

### The Master Weights Approach

```
Forward/Backward Pass (FP16/BF16):
    FP16 weights → forward → FP16 activations → loss
    loss → backward → FP16 gradients

Weight Update (FP32):
    FP32 master weights += optimizer_update(FP32 gradients)
    FP16 weights = cast(FP32 master weights)  # for next iteration
```

The FP32 master copy is essential because weight updates are often very small relative to the weight magnitudes. In FP16, an update of $10^{-5}$ to a weight of $1.0$ would be lost entirely (the next representable FP16 value after 1.0 is 1.0009765625). In FP32, the update is captured faithfully.

### Loss Scaling for FP16

When using FP16, many gradient values fall below the minimum representable range ($\approx 6 \times 10^{-5}$) and underflow to zero. **Loss scaling** addresses this by multiplying the loss by a large constant $S$ before the backward pass, which scales all gradients by $S$. After the backward pass, gradients are divided by $S$ before the optimizer step:

$$\text{scaled\_loss} = S \times \mathcal{L}$$
$$\text{scaled\_gradients} = \nabla_\theta (S \times \mathcal{L}) = S \times \nabla_\theta \mathcal{L}$$
$$\text{unscaled\_gradients} = \frac{\text{scaled\_gradients}}{S} = \nabla_\theta \mathcal{L}$$

**Dynamic loss scaling** starts with a large $S$ (e.g., $2^{16}$) and adjusts:
- If no overflow/NaN detected in gradients for $N$ consecutive steps: $S \leftarrow 2S$ (increase scale)
- If overflow/NaN detected: $S \leftarrow S/2$ (decrease scale), skip the optimizer step

### Why BF16 Does Not Need Loss Scaling

BF16 has the same exponent range as FP32, so gradients that are representable in FP32 will not underflow in BF16. The reduced mantissa precision (7 bits vs. 23 bits) means individual values are less precise, but this noise is well within the tolerance of stochastic gradient descent. This is why BF16 has become the preferred precision for modern training — it provides the memory and throughput benefits of half precision without the complexity of loss scaling.

### PyTorch Implementation

```python
import torch
from torch.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# ---- FP16 with loss scaling ----
scaler = GradScaler()

for inputs, targets in dataloader:
    inputs, targets = inputs.cuda(), targets.cuda()
    optimizer.zero_grad()

    # Forward pass in FP16
    with autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

    # Backward pass with scaled loss
    scaler.scale(loss).backward()

    # Unscale gradients, clip, step
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)  # Skips step if gradients contain inf/nan
    scaler.update()         # Adjust scale factor

# ---- BF16 without loss scaling (simpler) ----
for inputs, targets in dataloader:
    inputs, targets = inputs.cuda(), targets.cuda()
    optimizer.zero_grad()

    with autocast(device_type='cuda', dtype=torch.bfloat16):
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    # No scaler needed!
```

### What `autocast` Does

`torch.autocast` automatically selects the precision for each operation:

| Operation | Precision under autocast |
|---|---|
| Linear layers (`nn.Linear`) | FP16/BF16 |
| Convolutions (`nn.Conv2d`) | FP16/BF16 |
| Matrix multiplications (`torch.matmul`) | FP16/BF16 |
| Softmax | FP32 (for numerical stability) |
| Layer normalization | FP32 |
| Loss functions | FP32 |
| Batch normalization | FP32 |

Operations that are sensitive to precision (reductions, normalizations) remain in FP32, while compute-heavy operations (matrix multiplications) use reduced precision. This is handled automatically — the user does not need to manually cast tensors.

### Throughput and Memory Impact

On NVIDIA A100 (FP16 Tensor Core peak: 312 TFLOPS vs. FP32: 19.5 TFLOPS):
- **Throughput improvement:** 2-3x for training (theoretical 16x from Tensor Cores, limited by memory bandwidth and non-matmul operations)
- **Memory reduction:** ~50% for model weights and activations (from 4 bytes to 2 bytes per element)
- **Additional memory:** FP32 master weights add back $4\Psi$ bytes, but optimizer states were already FP32

---

## 13.3 Flash Attention — Versions 1, 2, and 3

Standard self-attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

where $Q, K, V \in \mathbb{R}^{N \times d}$ for sequence length $N$ and head dimension $d$. The intermediate matrix $QK^T \in \mathbb{R}^{N \times N}$ is the fundamental bottleneck.

### The Memory Problem

For sequence length $N = 8192$ and 32 attention heads in BF16:
$$\text{Attention matrix size} = N^2 \times 2 \text{ bytes} = 8192^2 \times 2 = 128 \text{ MB per head}$$
$$\text{Total across heads} = 128 \times 32 = 4 \text{ GB per layer}$$

This $O(N^2)$ memory requirement makes long-sequence training extremely expensive, and the memory is on HBM (high-bandwidth memory), which is the scarcest resource on a GPU.

### Flash Attention V1: The Key Insight

**Flash Attention** (Dao et al., 2022) recognizes that the bottleneck is not compute but **memory I/O**. The standard attention implementation reads and writes the $N \times N$ attention matrix from/to HBM, which is slow. The key insight is:

> *Never materialize the full $N \times N$ attention matrix in HBM. Instead, compute attention in tiles, keeping intermediate results in SRAM (on-chip cache).*

**GPU Memory Hierarchy:**
- **SRAM (on-chip):** ~20 MB on A100, ~19 TB/s bandwidth
- **HBM (off-chip):** 80 GB on A100, ~2 TB/s bandwidth

SRAM is ~10x faster than HBM but ~4000x smaller. Flash Attention tiles the computation to fit within SRAM.

### Tiling Algorithm

1. Divide $Q$ into blocks of size $B_r \times d$ and $K, V$ into blocks of size $B_c \times d$.
2. For each block of $Q$:
   a. Load $Q$ block into SRAM.
   b. For each block of $K, V$:
      - Load $K, V$ block into SRAM.
      - Compute $S_{ij} = Q_i K_j^T / \sqrt{d}$ in SRAM (size $B_r \times B_c$).
      - Compute local softmax and accumulate output using the **online softmax** trick.
   c. Write final output block to HBM.

The **online softmax** trick (Milakov & Gimelshein, 2018) maintains running statistics:

$$m_{\text{new}} = \max(m_{\text{old}}, \max(S_{ij}))$$
$$\ell_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} \cdot \ell_{\text{old}} + \sum_j e^{S_{ij} - m_{\text{new}}}$$
$$O_{\text{new}} = \frac{e^{m_{\text{old}} - m_{\text{new}}} \cdot \ell_{\text{old}} \cdot O_{\text{old}} + e^{S_{ij} - m_{\text{new}}} \cdot V_j}{\ell_{\text{new}}}$$

This allows computing the exact softmax without materializing the full row of attention scores.

### Complexity Analysis

| | Standard Attention | Flash Attention |
|---|---|---|
| HBM reads/writes | $O(N^2 + Nd)$ | $O(N^2 d / M)$ |
| Memory (HBM) | $O(N^2)$ | $O(N)$ |
| FLOPs | $O(N^2 d)$ | $O(N^2 d)$ (same) |

Where $M$ is the SRAM size. The FLOPs are identical — Flash Attention is not algorithmically faster in terms of operations. It is faster because it reduces HBM I/O by a factor of $d / \sqrt{M}$, which is significant (e.g., $d=128$, $M=100\text{KB}$ gives a ~40x I/O reduction).

**Practical speedup:** 2-4x wall-clock speedup on A100 for typical sequence lengths, with additional speedup from memory savings enabling larger batch sizes.

### Flash Attention V2

**Flash Attention 2** (Dao, 2023) introduced several optimizations:

1. **Better work partitioning:** The V1 algorithm parallelized over batch size and heads, leaving each thread block to handle the full sequence. V2 additionally parallelizes over the sequence length dimension, improving GPU occupancy for long sequences.

2. **Reduced non-matmul FLOPs:** V1 performed rescaling operations that, while asymptotically negligible, consumed significant wall-clock time (up to 50% on some architectures due to lower throughput of non-matmul operations). V2 restructures the computation to minimize these operations.

3. **Improved inner loop:** The outer loop iterates over K/V blocks and the inner loop over Q blocks (reversed from V1). This avoids the need to rescale the output and running softmax statistics in the inner loop.

**Result:** 1.5-2x speedup over Flash Attention V1, achieving 50-73% of theoretical maximum FLOPS on A100.

### Flash Attention V3

**Flash Attention 3** (Shah et al., 2024) targets the NVIDIA Hopper architecture (H100) and introduces:

1. **Asynchronous computation:** Exploits the Tensor Memory Accelerator (TMA) on H100 for asynchronous data movement, overlapping GEMM computations with softmax operations on different warps.

2. **Warp specialization:** Dedicates some warps to data loading and others to computation, achieving true overlap of memory access and compute.

3. **FP8 support:** Leverages FP8 Tensor Cores on H100 for up to 2x additional throughput, with careful handling of the reduced precision for attention computation.

**Result:** Up to 1.5-2x speedup over Flash Attention V2 on H100, reaching 75%+ utilization of H100 Tensor Cores.

### Using Flash Attention

```python
# Flash Attention is integrated into PyTorch via scaled_dot_product_attention
import torch
import torch.nn.functional as F

# PyTorch 2.0+ automatically uses Flash Attention when available
q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)
k = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)
v = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)

# This automatically selects Flash Attention or memory-efficient attention
output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# Or use the flash_attn package directly for more control
from flash_attn import flash_attn_func

# q, k, v shape: (batch, seqlen, nheads, headdim)
output = flash_attn_func(q, k, v, causal=True)
```

---

## 13.4 Quantization for Inference — GPTQ, AWQ, and GGUF

Post-training quantization reduces model size and inference cost by converting weights from FP16/BF16 to lower precision (INT8, INT4) after training is complete. The challenge is maintaining model quality despite the reduced precision.

### GPTQ: Optimal Post-Training Quantization

**GPTQ** (Frantar et al., 2022) is based on the Optimal Brain Quantization (OBQ) framework. The key idea is to quantize weights one at a time, optimally adjusting the remaining unquantized weights to compensate for the quantization error.

**Mathematical formulation:** For a weight matrix $W$ and a calibration set of inputs that produces activation matrix $X$, the goal is to find quantized weights $\hat{W}$ that minimize:

$$\min_{\hat{W}} \| WX - \hat{W}X \|_F^2$$

**OBQ algorithm (simplified):**

For each column $j$ of $W$:
1. Quantize weight $w_j$ to its nearest quantized value $\hat{w}_j$.
2. Compute the quantization error $\delta_j = w_j - \hat{w}_j$.
3. Update all remaining unquantized weights to compensate:
$$w_i \leftarrow w_i - \frac{\delta_j \cdot H_{ij}^{-1}}{H_{jj}^{-1}}$$

where $H = 2XX^T$ is the Hessian of the quantization objective (which is quadratic, so the Hessian is constant and can be precomputed).

**GPTQ's innovation** over vanilla OBQ is:
- **Block-wise quantization:** Process columns in blocks of 128, updating the Hessian only within blocks. This reduces computational cost from $O(d^3)$ to $O(d \cdot B^2)$ per row, where $B$ is the block size.
- **Lazy batch updates:** Accumulate weight adjustments and apply them in batches, improving GPU utilization.
- **Layer-by-layer:** Quantize one layer at a time, using the quantized layer's output as input to calibrate the next layer.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
quantization_config = GPTQConfig(
    bits=4,
    dataset="c4",           # Calibration dataset
    group_size=128,         # Quantize in groups of 128 weights
    desc_act=True,          # Order columns by activation magnitude
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto",
)
model.save_pretrained("llama-2-7b-gptq-4bit")
```

### AWQ: Activation-Aware Weight Quantization

**AWQ** (Lin et al., 2023) observes that not all weights are equally important — a small fraction of "salient" weights carry disproportionate importance for model quality. These salient weights correspond to large activation magnitudes.

**Key insight:** Instead of quantizing all weights uniformly, protect salient channels by scaling them up before quantization:

$$\hat{W} = \text{Quantize}(W \cdot \text{diag}(s))$$

where $s$ is a per-channel scaling vector chosen to minimize the quantization error for the most important channels. The inverse scaling $\text{diag}(s)^{-1}$ is absorbed into the preceding layer's output (or applied as a de-quantization step).

**Why AWQ outperforms naive quantization:**

Naive INT4 quantization assigns equal quantization resolution to all weights. But if a few channels have 100x larger activations than others, quantization error in those channels has a 100x larger impact on the output. AWQ effectively allocates more quantization levels to these critical channels.

**Scaling factor search:** AWQ searches for the optimal scale $s_j$ per channel by grid search over $\alpha \in [0, 1]$:

$$s_j = \left(\frac{\max(|X_j|)}{\max(|W_j|)}\right)^\alpha$$

The $\alpha$ that minimizes the output error on a calibration set is selected.

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-2-7b-hf"
quant_path = "llama-2-7b-awq-4bit"

model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.quantize(
    tokenizer,
    quant_config={
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",  # or "GEMV" for batch_size=1
    }
)

model.save_quantized(quant_path)
```

### GGUF Format

**GGUF** (GPT-Generated Unified Format) is the quantization format used by **llama.cpp** and its ecosystem. Unlike GPTQ and AWQ which produce GPU-optimized quantized models, GGUF is designed for efficient CPU inference (and can also run on GPUs via Metal or CUDA backends).

**Quantization levels:**

| Format | Bits/Weight | Method | Quality | Speed |
|---|---|---|---|---|
| Q2_K | 2.5 | K-quant, mixed 2/3-bit | Poor | Fastest |
| Q3_K_M | 3.4 | K-quant, mixed 3/4-bit | Fair | Very fast |
| Q4_0 | 4.0 | Symmetric, no scale opt | Good | Fast |
| Q4_K_M | 4.5 | K-quant, mixed 4/5-bit | Very good | Fast |
| Q5_K_M | 5.5 | K-quant, mixed 5/6-bit | Excellent | Medium |
| Q6_K | 6.5 | K-quant, 6-bit | Near-lossless | Medium |
| Q8_0 | 8.0 | Symmetric 8-bit | Lossless (practical) | Slow |

**K-quant** methods (Gerganov, 2023) use different bit widths for different layers based on their sensitivity. Attention layers and the first/last layers typically get higher precision, while MLP layers use lower precision.

**Size comparison for a 7B model:**

| Format | Model Size | RAM Required |
|---|---|---|
| FP16 | 13.5 GB | ~16 GB |
| Q8_0 | 7.2 GB | ~9.5 GB |
| Q5_K_M | 4.8 GB | ~7.1 GB |
| Q4_K_M | 4.1 GB | ~6.4 GB |
| Q4_0 | 3.8 GB | ~6.1 GB |
| Q2_K | 2.7 GB | ~5.0 GB |

### Choosing a Quantization Method

| Criterion | GPTQ | AWQ | GGUF |
|---|---|---|---|
| Target hardware | GPU | GPU | CPU (and GPU) |
| Quantization time | Minutes to hours | Minutes | Minutes |
| Quality at 4-bit | Good | Better | Good (K-quant) |
| Inference speed (GPU) | Fast | Fastest | Moderate |
| Ecosystem | HF Transformers, vLLM | HF Transformers, vLLM | llama.cpp, Ollama |
| Requires calibration data | Yes (128+ samples) | Yes (small set) | No (weight-only) |

---

## 13.5 Speculative Decoding

Autoregressive generation is inherently sequential: each token depends on all previous tokens. This means the GPU is severely underutilized during generation — each step performs a full forward pass but generates only a single token. **Speculative decoding** (Leviathan et al., 2023; Chen et al., 2023) breaks this bottleneck by parallelizing the verification of multiple candidate tokens.

### The Algorithm

1. **Draft phase:** A small, fast "draft" model $M_q$ generates $K$ candidate tokens autoregressively:
$$\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_K \sim M_q(y | x, y_{<t})$$

2. **Verification phase:** The large "target" model $M_p$ processes all $K$ candidates **in a single forward pass** (as a batch), computing the probability distribution at each position:
$$p_i = M_p(y | x, y_{<t}, \hat{y}_1, \ldots, \hat{y}_{i-1}) \quad \text{for } i = 1, \ldots, K$$

3. **Accept/reject:** For each candidate $\hat{y}_i$ in order:
   - If $p_i(\hat{y}_i) \geq q_i(\hat{y}_i)$: **accept** with probability 1.
   - If $p_i(\hat{y}_i) < q_i(\hat{y}_i)$: **accept** with probability $p_i(\hat{y}_i) / q_i(\hat{y}_i)$.
   - If rejected: sample a **correction token** from the adjusted distribution:
$$y_i \sim \text{norm}\left(\max(0, p_i - q_i)\right)$$
   - Stop processing further candidates after the first rejection.

### Mathematical Guarantee: Lossless Generation

The acceptance-rejection scheme guarantees that the output distribution is **exactly** the target model's distribution $M_p$, regardless of the quality of the draft model.

**Proof sketch:** For a candidate token $\hat{y}$ with draft probability $q(\hat{y})$ and target probability $p(\hat{y})$:

- The probability of accepting $\hat{y}$ is $\min(1, p(\hat{y})/q(\hat{y})) \cdot q(\hat{y}) = \min(q(\hat{y}), p(\hat{y}))$.
- The probability of rejection is $1 - \sum_y \min(q(y), p(y))$.
- Upon rejection, we sample from $\text{norm}(\max(0, p - q))$, which has probability mass $\sum_y \max(0, p(y) - q(y)) = 1 - \sum_y \min(q(y), p(y))$.

The total probability of producing token $y$ is:

$$P(y) = \min(q(y), p(y)) + \left(1 - \sum_{y'} \min(q(y'), p(y'))\right) \cdot \frac{\max(0, p(y) - q(y))}{\sum_{y'} \max(0, p(y') - q(y'))}$$

$$= \min(q(y), p(y)) + \max(0, p(y) - q(y)) = p(y) \quad \square$$

### Speedup Analysis

Let $\alpha$ be the average acceptance rate and $K$ be the number of draft tokens. The expected number of accepted tokens per verification step is:

$$\mathbb{E}[\text{accepted tokens}] = \frac{1 - \alpha^{K+1}}{1 - \alpha}$$

For $\alpha = 0.8$ and $K = 5$: $\mathbb{E} \approx 4.0$ tokens per step, yielding a ~4x speedup (minus the draft model's cost).

**Practical speedup** is typically 2-3x because:
- The draft model has non-zero cost (typically 5-15% of the target model).
- The verification pass processes $K$ tokens (more compute than a single-token forward pass, though much less than $K$ separate passes due to parallelism).
- Acceptance rates vary by task and temperature.

### Implementation Considerations

```python
# Conceptual implementation of speculative decoding
def speculative_decode(target_model, draft_model, prompt, K=5, max_tokens=100):
    generated = list(prompt)

    while len(generated) - len(prompt) < max_tokens:
        # Draft phase: generate K candidates
        draft_tokens = []
        draft_probs = []
        draft_input = generated.copy()

        for _ in range(K):
            logits = draft_model(draft_input)
            prob = torch.softmax(logits[-1], dim=-1)
            token = torch.multinomial(prob, 1).item()
            draft_tokens.append(token)
            draft_probs.append(prob)
            draft_input.append(token)

        # Verification phase: single forward pass through target model
        verify_input = generated + draft_tokens
        target_logits = target_model(verify_input)

        # Accept/reject
        n_accepted = 0
        for i in range(K):
            pos = len(generated) + i - 1
            target_prob = torch.softmax(target_logits[pos], dim=-1)
            q = draft_probs[i][draft_tokens[i]]
            p = target_prob[draft_tokens[i]]

            # Acceptance criterion
            if torch.rand(1).item() < (p / q).clamp(max=1.0).item():
                generated.append(draft_tokens[i])
                n_accepted += 1
            else:
                # Sample correction token
                adjusted = torch.clamp(target_prob - draft_probs[i], min=0)
                adjusted = adjusted / adjusted.sum()
                token = torch.multinomial(adjusted, 1).item()
                generated.append(token)
                break

        # If all K accepted, sample one more from target
        if n_accepted == K:
            target_prob = torch.softmax(target_logits[len(generated) - 1], dim=-1)
            token = torch.multinomial(target_prob, 1).item()
            generated.append(token)

    return generated
```

### Draft Model Selection

The draft model should be:
- **Fast:** 10-20x fewer parameters than the target (e.g., 68M draft for a 7B target).
- **Aligned:** Trained on similar data or distilled from the target for high acceptance rates.
- **Same tokenizer:** Must use the identical vocabulary and tokenization.

Common pairings:
- LLaMA 7B + LLaMA 68M (purpose-trained draft)
- GPT-4 + GPT-3.5 (same family)
- Any model + n-gram model or retrieval-based draft

---

## 13.6 PagedAttention and vLLM Memory Management

During autoregressive generation, the **KV cache** stores the key and value tensors from all previous tokens so they do not need to be recomputed. This cache is the primary memory bottleneck during LLM serving.

### The KV Cache Problem

For a model with $L$ layers, $H$ attention heads, head dimension $d_h$, and sequence length $s$ in BF16:

$$\text{KV cache size} = 2 \times L \times H \times d_h \times s \times 2 \text{ bytes}$$

For LLaMA-2 70B ($L=80$, $H=64$, $d_h=128$) with $s=4096$:

$$\text{KV cache} = 2 \times 80 \times 64 \times 128 \times 4096 \times 2 = 10.7 \text{ GB per request}$$

With naive memory allocation, serving even a handful of concurrent requests exhausts GPU memory. Moreover, because sequence lengths are variable (requests may generate anywhere from 1 to 4096 tokens), naive pre-allocation wastes enormous amounts of memory through **internal fragmentation**.

### PagedAttention: Virtual Memory for KV Cache

**PagedAttention** (Kwon et al., 2023) draws inspiration from operating system virtual memory to solve this problem.

**Key concepts:**

1. **Blocks:** The KV cache is divided into fixed-size blocks (e.g., 16 tokens per block). Each block stores the key and value tensors for 16 consecutive tokens across all layers and heads.

2. **Page table:** Each request has a logical-to-physical page table that maps logical blocks (positions in the sequence) to physical blocks (locations in GPU memory).

3. **Non-contiguous allocation:** Physical blocks can be scattered anywhere in GPU memory. The page table handles the mapping, so no contiguous memory allocation is needed.

4. **On-demand allocation:** New blocks are allocated only when a request generates more tokens — no pre-allocation of maximum sequence length.

```
Traditional KV Cache:
Request 1: [████████████████████████████████████████░░░░░░░░░░]  (60% used, 40% wasted)
Request 2: [████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  (40% used, 60% wasted)

PagedAttention:
Physical blocks: [R1][R2][R1][R1][R2][R1][free][R2][R1][free][R2]...
Page Table R1:    logical 0→phys 0, 1→2, 2→3, 3→5, 4→8
Page Table R2:    logical 0→phys 1, 1→4, 2→7, 3→10
```

### Copy-on-Write for Beam Search

When multiple beams share a prefix (which is common), PagedAttention uses **copy-on-write** semantics: shared blocks are reference-counted and only copied when a beam diverges. This reduces memory usage for beam search by up to 55%.

### Memory Efficiency

PagedAttention achieves near-optimal memory utilization:
- **Internal fragmentation:** Limited to the last block of each request (~8 tokens on average with block size 16).
- **External fragmentation:** Zero — blocks can be allocated from any free location.
- **Waste reduction:** From 60-80% waste (pre-allocation) to <4% waste.

**Practical impact:** vLLM serves 2-4x more concurrent requests than HuggingFace TGI and FasterTransformer by making better use of the same GPU memory (Kwon et al., 2023).

---

## 13.7 Continuous Batching for Serving

### Static Batching

Traditional inference servers batch requests together and process the entire batch synchronously:

```
Time →
Batch of 4 requests:
Request 1: [prompt→→→→→→→ | gen gen gen gen gen DONE]
Request 2: [prompt→→→→→→→ | gen gen gen gen gen gen gen gen gen DONE]
Request 3: [prompt→→→→→→→ | gen gen DONE                          ]
Request 4: [prompt→→→→→→→ | gen gen gen gen gen gen DONE           ]
                                                     ↑ All wait for longest request
```

Request 3 finishes early but its GPU resources are wasted until Request 2 (the longest) finishes. The GPU utilization drops as more requests complete.

### Continuous Batching (Iteration-Level Scheduling)

**Continuous batching** (Yu et al., 2022) makes scheduling decisions at each **generation step** rather than at the batch level:

```
Time →
Step 1: [R1, R2, R3, R4] — all active
Step 2: [R1, R2, R3, R4] — all active
Step 3: [R1, R2, R4, R5] — R3 done, R5 admitted   ← new request fills the slot
Step 4: [R1, R2, R4, R5] — all active
Step 5: [R2, R4, R5, R6] — R1 done, R6 admitted
...
```

When a request finishes generating, its slot is immediately filled by a new request from the queue. This keeps the GPU continuously saturated.

### Implementation in vLLM and TGI

**vLLM** implements continuous batching with the following components:

1. **Scheduler:** At each step, decides which requests to process (may include prefill for new requests and decode for existing ones).
2. **Block manager:** Allocates PagedAttention blocks for new requests and reclaims blocks from completed requests.
3. **Request queue:** New requests wait in a FIFO queue and are admitted when GPU memory is available.

**TGI (Text Generation Inference)** by Hugging Face implements a similar scheme called **in-flight batching** with configurable maximum batch size and token budget.

### Throughput Impact

Continuous batching improves throughput by:
- **Eliminating idle time:** GPUs are always processing the maximum number of requests.
- **Better memory utilization:** Memory from completed requests is immediately reclaimed.
- **Adaptive scheduling:** The batch can grow or shrink based on current load.

Typical improvement: **2-8x throughput increase** over static batching, depending on the variance in output sequence lengths (higher variance = more benefit).

```python
# Conceptual continuous batching loop
class ContinuousBatcher:
    def __init__(self, model, max_batch_size, max_tokens):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_tokens = max_tokens
        self.active_requests = []
        self.request_queue = []

    def step(self):
        # Remove completed requests
        self.active_requests = [
            r for r in self.active_requests if not r.is_finished()
        ]

        # Admit new requests from queue
        while (len(self.active_requests) < self.max_batch_size
               and self.request_queue
               and self.has_memory_for_new_request()):
            new_request = self.request_queue.pop(0)
            self.active_requests.append(new_request)

        if not self.active_requests:
            return

        # Run one decode step for all active requests
        batch = self.prepare_batch(self.active_requests)
        next_tokens = self.model.generate_step(batch)

        # Update each request
        for request, token in zip(self.active_requests, next_tokens):
            request.append_token(token)
            if token == EOS_TOKEN or request.length >= request.max_length:
                request.mark_finished()
                self.return_result(request)
```

---

## 13.8 KV Cache Optimization

Beyond PagedAttention and continuous batching, several techniques specifically target KV cache size reduction.

### Prefix Caching

Many applications share a common system prompt across requests (e.g., "You are a helpful assistant..."). **Prefix caching** computes the KV cache for the shared prefix once and reuses it across requests:

```
System prompt KV cache: [cached once in GPU memory]
                              ↓
Request 1: [cached prefix] + [user query 1 KV] → generation
Request 2: [cached prefix] + [user query 2 KV] → generation
Request 3: [cached prefix] + [user query 3 KV] → generation
```

With PagedAttention, prefix caching is implemented by sharing the physical blocks of the prefix across request page tables (using reference counting).

**Memory savings:** For a 2000-token system prompt and 100 concurrent requests, prefix caching saves $99 \times 2000 \times \text{KV\_size\_per\_token}$, which for a 70B model is approximately $99 \times 2000 \times 2.6\text{ KB} = 500\text{ MB}$.

vLLM implements **automatic prefix caching** that detects shared prefixes across requests and caches them transparently.

### Multi-Query Attention (MQA)

**Multi-Query Attention** (Shazeer, 2019) uses multiple query heads but **a single key and value head**:

- Standard MHA: $H$ query heads, $H$ key heads, $H$ value heads
- MQA: $H$ query heads, **1** key head, **1** value head

KV cache reduction: $H\times$ smaller. For $H = 64$, this is a **64x** reduction in KV cache size.

**Trade-off:** Some quality degradation compared to full MHA, especially for smaller models.

### Grouped-Query Attention (GQA)

**Grouped-Query Attention** (Ainslie et al., 2023) is a compromise between MHA and MQA. It uses $G$ groups, where each group of $H/G$ query heads shares one key and one value head:

- MHA: $G = H$ (every query head has its own KV)
- MQA: $G = 1$ (all query heads share one KV)
- GQA: $1 < G < H$ (intermediate)

LLaMA 2 70B uses GQA with $G = 8$ (8 KV heads for 64 query heads), reducing KV cache by 8x with minimal quality loss.

```
Full MHA (H=8):  Q₁K₁V₁  Q₂K₂V₂  Q₃K₃V₃  Q₄K₄V₄  Q₅K₅V₅  Q₆K₆V₆  Q₇K₇V₇  Q₈K₈V₈
GQA (G=2):       Q₁Q₂Q₃Q₄→K₁V₁    Q₅Q₆Q₇Q₈→K₂V₂
MQA (G=1):       Q₁Q₂Q₃Q₄Q₅Q₆Q₇Q₈→K₁V₁
```

### KV Cache Compression

Beyond architectural changes, several post-hoc compression techniques reduce KV cache size:

1. **Quantized KV cache:** Store keys and values in INT8 or INT4 instead of FP16/BF16. Typically maintains quality because attention patterns are robust to KV quantization. 2x-4x memory reduction.

2. **Token eviction:** Only keep KV entries for the most "important" tokens (e.g., those with highest attention weights). Techniques like **H2O (Heavy-Hitter Oracle)** (Zhang et al., 2024) identify and retain only the tokens that receive the most attention.

3. **KV cache merging:** Compress consecutive KV entries into a smaller representation, similar to pooling in convolutional networks.

```python
# INT8 KV cache quantization in vLLM
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-70b-chat-hf",
    kv_cache_dtype="fp8",  # or "int8" — reduces KV cache memory by 2x
    tensor_parallel_size=4,
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
outputs = llm.generate(["Tell me about quantum computing."], sampling_params)
```

### Putting It All Together: Memory Budget for Serving

For a LLaMA-2 70B model serving concurrent requests on 4x A100 80GB GPUs:

| Component | Memory | Notes |
|---|---|---|
| Model weights (INT4) | ~35 GB | 4-bit quantized, split across 4 GPUs |
| KV cache pool | ~240 GB | Shared across all 4 GPUs |
| CUDA overhead | ~4 GB | Per GPU |
| **Total available** | **320 GB** | 4 × 80 GB |
| **Available for KV** | **~240 GB** | 320 - 35 - 16 (overhead) |

With GQA and FP8 KV cache:
- Per-token KV size: $2 \times 80 \times 8 \times 128 \times 1 = 160$ KB
- Maximum tokens in cache: $240 \text{ GB} / 160 \text{ KB} \approx 1.5M$ tokens
- At 4096 tokens per request: ~375 concurrent requests

Without GQA or FP8:
- Per-token KV size: $2 \times 80 \times 64 \times 128 \times 2 = 2.5$ MB
- Maximum tokens: ~100K tokens → ~24 concurrent requests

The combination of GQA + FP8 KV cache enables **15x more concurrent requests**.

---

## Exercises

1. **Precision Analysis:** A gradient value of $3.2 \times 10^{-6}$ needs to be represented. (a) Can FP16 represent this value? (b) What is the closest FP16 value? (c) Can BF16 represent this value? Explain the implications for training.

2. **Flash Attention Memory:** For a model with sequence length $N = 32768$, 32 attention heads, and head dimension $d = 128$: (a) Calculate the attention matrix memory in standard attention (BF16). (b) Calculate the memory usage with Flash Attention. (c) What is the memory reduction factor?

3. **Quantization Comparison:** You need to deploy a 13B parameter model on a single GPU with 24 GB of memory. Calculate the model size in: (a) FP16, (b) INT8, (c) INT4. Which quantization methods (GPTQ, AWQ, GGUF) would you consider and why?

4. **Speculative Decoding:** A draft model has an 85% acceptance rate when generating code. (a) With $K = 8$ draft tokens, what is the expected number of accepted tokens per verification? (b) If the draft model takes 2ms per token and the target model takes 50ms for a batch of 8 tokens, what is the effective latency per token? Compare to naive autoregressive generation.

5. **KV Cache Sizing:** For a model with 32 layers, 32 attention heads (GQA with 4 KV heads), head dimension 128, serving with FP8 KV cache: (a) Calculate the KV cache size per token. (b) If you have 40 GB available for KV cache, how many tokens can you store? (c) How many concurrent 8K-context requests can you serve?

6. **Implementation:** Implement a simple PagedAttention-like memory manager in Python that allocates fixed-size blocks, maintains per-request page tables, and reclaims blocks when requests complete. Demonstrate with a simulation of 10 concurrent requests with varying sequence lengths.

7. **Mixed Precision Training:** Write a training loop that uses BF16 autocast, gradient accumulation with 4 steps, and gradient clipping. Include proper handling of the `no_sync()` context for DDP. Verify that the effective gradient matches a single large-batch computation.

---

## References

Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., & Sanghai, S. (2023). GQA: Training generalized multi-query transformer models from multi-head checkpoints. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, 4895-4901.

Chen, C., Borgeaud, S., Irving, G., Lespiau, J. B., Sifre, L., & Jumper, J. (2023). Accelerating large language model decoding with speculative sampling. *arXiv preprint arXiv:2302.01318*.

Dao, T. (2023). FlashAttention-2: Faster attention with better parallelism and work partitioning. *arXiv preprint arXiv:2307.08691*.

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. *Advances in Neural Information Processing Systems*, 35, 16344-16359.

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized language models. *Advances in Neural Information Processing Systems*, 36.

Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2022). GPTQ: Accurate post-training quantization for generative pre-trained transformers. *arXiv preprint arXiv:2210.17323*.

Gerganov, G. (2023). llama.cpp k-quants. *GitHub: ggerganov/llama.cpp*.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., ... & Stoica, I. (2023). Efficient memory management for large language model serving with PagedAttention. *Proceedings of the 29th Symposium on Operating Systems Principles*, 611-626.

Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast inference from transformers via speculative decoding. *International Conference on Machine Learning*, 19274-19286.

Lin, J., Tang, J., Tang, H., Yang, S., Dang, X., & Han, S. (2023). AWQ: Activation-aware weight quantization for LLM compression and acceleration. *arXiv preprint arXiv:2306.00978*.

Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., ... & Wu, H. (2018). Mixed precision training. *International Conference on Learning Representations*.

Milakov, M., & Gimelshein, N. (2018). Online normalizer calculation for softmax. *arXiv preprint arXiv:1805.02867*.

Shah, J., Dao, T., et al. (2024). FlashAttention-3: Fast and accurate attention with asynchrony and low-precision. *arXiv preprint arXiv:2407.08691*.

Shazeer, N. (2019). Fast transformer decoding: One write-head is all you need. *arXiv preprint arXiv:1911.02150*.

Yu, G. I., Jeong, J. S., Kim, G. W., Kim, S., & Chun, B. G. (2022). Orca: A distributed serving system for transformer-based generative models. *16th USENIX Symposium on Operating Systems Design and Implementation*, 521-538.

Zhang, Z., Sheng, Y., Zhou, T., Chen, T., Zheng, L., Cai, R., ... & Stoica, I. (2024). H2O: Heavy-hitter oracle for efficient generative inference of large language models. *Advances in Neural Information Processing Systems*, 36.
