# Chapter 14: Profiling and Performance Tuning

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Describe the architecture of modern NVIDIA GPUs including streaming multiprocessors, Tensor Cores, and the memory hierarchy.
2. Distinguish between compute-bound and memory-bound operations using arithmetic intensity and the roofline model.
3. Calculate Model FLOP Utilization (MFU) and interpret what constitutes good utilization for different training scenarios.
4. Use PyTorch Profiler to identify performance bottlenecks in training and inference workloads.
5. Navigate NVIDIA Nsight Systems and Nsight Compute to analyze GPU kernel performance.
6. Write custom Triton kernels for fused operations and understand the performance benefits of kernel fusion.
7. Understand CUDA programming fundamentals including the thread hierarchy, memory model, and common optimization patterns.
8. Apply a systematic methodology for end-to-end performance optimization of a training pipeline.

---

## 14.1 Understanding GPU Architecture

To optimize deep learning workloads, we must first understand the hardware we are optimizing for. Modern NVIDIA GPUs are massively parallel processors designed around a hierarchy of compute units and memory systems.

### Streaming Multiprocessors (SMs)

The GPU is organized into an array of **Streaming Multiprocessors (SMs)**. Each SM is an independent processing unit containing:

- **CUDA Cores:** General-purpose ALUs for FP32/FP64/INT32 operations. An A100 has 6912 CUDA cores across 108 SMs (64 per SM). An H100 has 16896 CUDA cores across 132 SMs (128 per SM).

- **Tensor Cores:** Specialized matrix-multiply-accumulate (MMA) units. An A100 has 432 third-generation Tensor Cores (4 per SM). An H100 has 528 fourth-generation Tensor Cores (4 per SM). Tensor Cores perform $D = A \times B + C$ on small matrix tiles (e.g., 16x16x16 in FP16) in a single clock cycle.

- **Register file:** 256 KB per SM — the fastest memory available to threads. Each thread gets a portion of this register file.

- **Shared memory / L1 cache:** Configurable, up to 228 KB per SM on H100. Shared among all threads within a thread block. Programmer-managed (shared memory) or hardware-managed (L1 cache).

- **Warp schedulers:** Each SM has 4 warp schedulers, each capable of issuing one instruction per cycle for its assigned warps.

### Memory Hierarchy

Understanding the memory hierarchy is critical for performance optimization:

```
Register File (per SM):     256 KB    |    ~40 TB/s    |   1 cycle latency
        ↓
Shared Memory / L1 (per SM): 228 KB  |    ~19 TB/s    |   ~30 cycles
        ↓
L2 Cache (shared):          40-50 MB  |    ~6 TB/s     |   ~200 cycles
        ↓
HBM (global memory):       80 GB      |    2-3.35 TB/s |   ~400 cycles
```

**Key insight:** The ratio between compute throughput and memory bandwidth defines the performance characteristics of the GPU. For H100:

- Peak FP16 Tensor Core throughput: 989 TFLOPS
- HBM bandwidth: 3.35 TB/s
- **Compute-to-memory ratio:** $989 \times 10^{12} / (3.35 \times 10^{12}) \approx 295$ FLOP/byte

This means the GPU can perform ~295 floating-point operations for every byte read from HBM. If an operation performs fewer than 295 FLOPs per byte of data accessed, it is **memory-bound** — the GPU will spend more time waiting for data than computing.

### NVIDIA A100 vs. H100 Specifications

| Specification | A100 (SXM) | H100 (SXM) |
|---|---|---|
| Architecture | Ampere (GA100) | Hopper (GH100) |
| SMs | 108 | 132 |
| CUDA Cores | 6912 | 16896 |
| Tensor Cores | 432 (3rd gen) | 528 (4th gen) |
| FP32 Peak | 19.5 TFLOPS | 67 TFLOPS |
| FP16 Tensor Core Peak | 312 TFLOPS | 989 TFLOPS |
| BF16 Tensor Core Peak | 312 TFLOPS | 989 TFLOPS |
| FP8 Tensor Core Peak | N/A | 1979 TFLOPS |
| HBM | 80 GB HBM2e | 80 GB HBM3 |
| Memory Bandwidth | 2.0 TB/s | 3.35 TB/s |
| NVLink Bandwidth | 600 GB/s | 900 GB/s |
| TDP | 400W | 700W |

The H100 represents a **3.2x improvement** in Tensor Core throughput over A100 and a **1.7x improvement** in memory bandwidth. The compute-to-bandwidth ratio increased from ~156 to ~295, meaning more operations are memory-bound on H100 than on A100.

### Tensor Core Operation

Tensor Cores perform a matrix multiply-accumulate on small tile sizes in a single instruction:

```
D[M×N] = A[M×K] × B[K×N] + C[M×N]
```

Tile sizes (H100, 4th generation):
- FP16: 16×16×16
- BF16: 16×16×16
- FP8: 16×16×32
- INT8: 16×16×32

To achieve peak Tensor Core utilization, matrix dimensions must be multiples of these tile sizes (or at least multiples of 8/16). Odd dimensions cause wasted compute.

---

## 14.2 Compute-Bound vs. Memory-Bound Operations

### Arithmetic Intensity

**Arithmetic intensity** (also called operational intensity) is the ratio of floating-point operations to bytes transferred:

$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Accessed}}$$

This metric determines whether an operation is compute-bound or memory-bound.

### The Roofline Model

The **roofline model** (Williams et al., 2009) relates achievable performance to arithmetic intensity:

$$\text{Achievable FLOP/s} = \min\left(\text{Peak FLOP/s}, \quad \text{Bandwidth} \times \text{Arithmetic Intensity}\right)$$

```
Performance (FLOP/s)
    │
    │                        ╱─────────── Peak Compute (989 TFLOPS for H100)
    │                      ╱
    │                    ╱
    │                  ╱
    │                ╱  ← Ridge point (~295 FLOP/byte for H100)
    │              ╱
    │            ╱   ← Memory-bound region
    │          ╱      (slope = bandwidth)
    │        ╱
    │      ╱
    │    ╱
    │  ╱
    │╱
    └──────────────────────── Arithmetic Intensity (FLOP/byte)
```

Operations to the **left of the ridge point** are memory-bound (limited by how fast data can be delivered to the compute units). Operations to the **right** are compute-bound (limited by how fast the compute units can process data).

### Where Common Operations Fall

**GEMM (General Matrix Multiply):** $C = A \times B$ where $A \in \mathbb{R}^{M \times K}$ and $B \in \mathbb{R}^{K \times N}$.

- FLOPs: $2MKN$
- Bytes: $(MK + KN + MN) \times \text{bytes\_per\_element}$
- Arithmetic intensity: $\frac{2MKN}{(MK + KN + MN) \times b} \approx \frac{2N}{3b}$ for square matrices

For $M = K = N = 4096$ in FP16 ($b = 2$):
$$\text{AI} \approx \frac{2 \times 4096}{3 \times 2} \approx 1365 \text{ FLOP/byte}$$

This is **well above the ridge point** — GEMM is compute-bound for reasonably sized matrices. This is why Tensor Cores (which accelerate matrix multiplications) provide such dramatic speedups for deep learning.

**Element-wise operations** (ReLU, GELU, add, multiply): 1 FLOP per element, but must read and write each element (2-4 bytes). Arithmetic intensity: $\sim 0.25$ FLOP/byte. **Heavily memory-bound.**

**Softmax:** For a vector of length $N$: ~$5N$ FLOPs (compute max, subtract, exponentiate, sum, divide) and ~$4N$ bytes transferred. AI $\approx 1.25$ FLOP/byte. **Memory-bound.**

**Layer Normalization:** Similar to softmax — reduction operations followed by element-wise scaling. AI $\approx 1$ FLOP/byte. **Memory-bound.**

**Attention (standard):** $O(N^2 d)$ FLOPs for $O(N^2 + Nd)$ bytes. For long sequences where $N \gg d$, the attention matrix dominates and AI approaches $\sim d$ FLOP/byte — but the large intermediate matrix ($N^2$) must transit through HBM, making it memory-bound in practice. This is precisely why Flash Attention (which avoids materializing this matrix) provides speedups.

### Implications for Optimization

1. **Compute-bound operations:** Optimize by using higher-throughput instructions (Tensor Cores), reducing FLOPs (algorithm changes), or overlapping compute with communication.

2. **Memory-bound operations:** Optimize by reducing data movement. Key techniques:
   - **Kernel fusion:** Combine multiple memory-bound operations into a single kernel, eliminating intermediate writes to HBM.
   - **Operator reordering:** Batch operations that access the same data together.
   - **Quantization:** Reduce bytes per element.
   - **Flash Attention:** Avoid materializing intermediate tensors.

---

## 14.3 Model FLOP Utilization (MFU)

### Definition

**Model FLOP Utilization (MFU)** measures what fraction of a system's peak FLOP/s is actually used for useful model computation (Chowdhery et al., 2022):

$$\text{MFU} = \frac{\text{Model FLOPs per step} / \text{Time per step}}{\text{Peak System FLOP/s}}$$

MFU intentionally counts only the **model's theoretical FLOPs** (forward + backward pass computations), not overhead like communication, data loading, or rematerialization from gradient checkpointing.

### Calculating Model FLOPs

For a Transformer-based language model with:
- $L$ layers
- Hidden dimension $h$
- Sequence length $s$
- Batch size $B$ (in sequences)
- Vocabulary size $V$
- Using BF16 training

**FLOPs per token (forward pass):**

The dominant operations are the linear projections in attention and MLP:

$$\text{FLOPs}_{\text{forward}} \approx 2 \times P \times s \times B$$

where $P$ is the number of model parameters (since each parameter participates in one multiply-add per token in the forward pass). The factor of 2 accounts for the multiply and add operations in matrix multiplications.

More precisely, for a standard transformer:

| Operation | FLOPs per token |
|---|---|
| QKV projection | $6 \times h^2$ |
| Attention logits | $2 \times s \times h$ |
| Attention × V | $2 \times s \times h$ |
| Output projection | $2 \times h^2$ |
| MLP (up + down) | $16 \times h^2$ (with 4h intermediate) |
| **Total per layer** | **$24h^2 + 4sh$** |
| **Total model** | **$L(24h^2 + 4sh) + 2hV$** |

For a 7B model ($L=32$, $h=4096$, $s=2048$, $V=32000$):
$$\text{FLOPs}_{\text{per token}} \approx 32 \times (24 \times 4096^2 + 4 \times 2048 \times 4096) + 2 \times 4096 \times 32000$$
$$\approx 13.7 \times 10^9 \approx 2 \times P$$

**Backward pass** is approximately $2\times$ the forward pass, giving a total of $\approx 6P$ FLOPs per token per training step.

### What is "Good" MFU?

| MFU | Assessment | Typical Scenario |
|---|---|---|
| > 60% | Excellent | Highly optimized, large batch, fast interconnect |
| 50-60% | Very good | Well-tuned production training |
| 40-50% | Good | Typical large-scale training |
| 30-40% | Fair | Room for optimization |
| < 30% | Poor | Significant bottlenecks |

**Notable MFU achievements:**
- **PaLM (540B):** 46.2% on TPUv4 pods (Chowdhery et al., 2022)
- **Megatron-LM (530B):** ~30% on A100 clusters (Smith et al., 2022)
- **LLaMA-1 (65B):** ~45% on A100 clusters (Touvron et al., 2023)

### Factors that Reduce MFU

1. **Communication overhead:** All-reduce, all-gather, reduce-scatter operations occupy the GPU without contributing model FLOPs.
2. **Pipeline bubbles:** Idle time in pipeline parallelism.
3. **Activation recomputation:** Gradient checkpointing recomputes forward activations during backward, adding FLOPs not counted in the "model FLOPs" (since they are recomputation, not unique model operations). This is why MFU can appear to decrease when enabling gradient checkpointing, even though throughput may increase (due to larger batch sizes fitting in memory).
4. **Non-Tensor-Core operations:** Softmax, layer norm, and other non-matmul operations do not utilize Tensor Cores and have much lower throughput.
5. **Data loading:** If the data pipeline cannot keep up with GPU compute, the GPU idles waiting for data.
6. **Small batch sizes:** Insufficient parallelism to saturate all SMs and Tensor Cores.
7. **Suboptimal matrix dimensions:** Dimensions that are not multiples of Tensor Core tile sizes waste compute.

### Computing MFU in Practice

```python
import time
import torch

def compute_mfu(
    model_params: int,
    batch_size: int,
    seq_length: int,
    step_time_seconds: float,
    num_gpus: int,
    peak_flops_per_gpu: float,  # e.g., 312e12 for A100 BF16
    gradient_checkpointing: bool = False,
):
    """
    Compute Model FLOP Utilization.

    Model FLOPs = 6 * params * tokens_per_step (forward + backward)
    Note: gradient checkpointing adds extra forward FLOPs,
    but MFU convention excludes recomputation.
    """
    tokens_per_step = batch_size * seq_length
    model_flops_per_step = 6 * model_params * tokens_per_step

    # Achieved FLOP/s
    achieved_flops = model_flops_per_step / step_time_seconds

    # Peak system FLOP/s
    peak_system_flops = num_gpus * peak_flops_per_gpu

    mfu = achieved_flops / peak_system_flops

    print(f"Tokens per step: {tokens_per_step:,}")
    print(f"Model FLOPs per step: {model_flops_per_step:.2e}")
    print(f"Achieved FLOP/s: {achieved_flops:.2e}")
    print(f"Peak system FLOP/s: {peak_system_flops:.2e}")
    print(f"MFU: {mfu:.1%}")

    return mfu

# Example: LLaMA 7B on 8x A100
mfu = compute_mfu(
    model_params=7e9,
    batch_size=32,
    seq_length=2048,
    step_time_seconds=2.5,
    num_gpus=8,
    peak_flops_per_gpu=312e12,  # A100 BF16 Tensor Core peak
)
# Output: MFU: ~35.1%
```

---

## 14.4 PyTorch Profiler — A Complete Guide

The PyTorch Profiler is the primary tool for identifying performance bottlenecks in PyTorch code. It captures CPU and GPU activity, memory allocation, and operator-level timing.

### Basic Usage

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule

model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
dataloader = get_dataloader()

# Define profiling schedule
# Wait 1 step (skip warmup), warm up for 1 step, profile 3 steps, repeat 1 time
prof_schedule = schedule(
    wait=1,
    warmup=1,
    active=3,
    repeat=1,
)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=prof_schedule,
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True,
) as prof:
    for step, (inputs, targets) in enumerate(dataloader):
        if step >= (1 + 1 + 3):  # wait + warmup + active
            break

        inputs = inputs.cuda()
        targets = targets.cuda()

        with record_function("forward"):
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        with record_function("backward"):
            optimizer.zero_grad()
            loss.backward()

        with record_function("optimizer"):
            optimizer.step()

        prof.step()  # Signal end of profiling step

# Print summary
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

### Understanding the Output

The profiler table shows:

```
---------------------------------  ------------  ------------  ------------  ------------
                             Name    Self CPU %    Self CUDA %     CUDA total      # Calls
---------------------------------  ------------  ------------  ------------  ------------
                          forward       15.2%         45.3%        234.5ms           3
                         backward       12.8%         38.7%        200.1ms           3
                        optimizer        3.1%          8.2%         42.3ms           3
                aten::mm              8.5%         32.1%        165.8ms          96
                aten::addmm           4.2%         15.3%         79.0ms          48
         aten::scaled_dot_product    2.1%         12.5%         64.5ms          48
                 aten::layer_norm      1.5%          3.2%         16.5ms          96
                      aten::gelu       0.8%          2.1%         10.8ms          48
---------------------------------  ------------  ------------  ------------  ------------
```

**Key metrics to examine:**

1. **CUDA time total:** Total GPU time including nested operations. Look for operations consuming disproportionate time.
2. **Self CUDA time:** GPU time excluding nested operations. High self-time indicates the operation itself is slow (not its children).
3. **CPU time:** If CPU time significantly exceeds CUDA time for an operation, the GPU may be idle waiting for CPU computation.
4. **Number of calls:** Unexpected call counts may indicate redundant computation.

### TensorBoard Visualization

```bash
tensorboard --logdir ./profiler_logs
```

The TensorBoard plugin provides:

1. **Overview:** High-level summary of time distribution across GPU kernel categories (computation, communication, memory, idle).
2. **Operator view:** Table of all operators sorted by GPU time.
3. **GPU kernel view:** Individual CUDA kernel performance.
4. **Trace view:** Chrome trace timeline showing CPU and GPU activity aligned in time. This is the most powerful view for identifying gaps and overlap opportunities.
5. **Memory view:** Memory allocation timeline showing peak usage and allocation patterns.

### Identifying Common Bottlenecks

**Bottleneck 1: CPU-side overhead (Python, data loading)**

Symptom: Large gaps between GPU kernels in the trace view. The GPU is idle waiting for the CPU to launch the next kernel.

```python
# Diagnosis: Check if data loading is the bottleneck
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("data_loading"):
        batch = next(iter(dataloader))
    with record_function("to_device"):
        batch = {k: v.cuda() for k, v in batch.items()}
    with record_function("forward"):
        outputs = model(**batch)
```

**Fix:** Increase DataLoader `num_workers`, use `pin_memory=True`, use `non_blocking=True` for `.cuda()` calls.

**Bottleneck 2: Synchronization points**

Symptom: `cudaDeviceSynchronize` or `torch.cuda.synchronize()` calls appearing in the trace, forcing the CPU to wait for all GPU work to complete.

**Fix:** Remove explicit synchronization. Use `torch.cuda.Event` for timing instead of `time.time()` with synchronization.

**Bottleneck 3: Small kernels with high launch overhead**

Symptom: Many tiny GPU kernels with gaps between them. The GPU launch latency (~5-10 microseconds) becomes dominant.

**Fix:** Use `torch.compile` to fuse small operations, or write custom Triton kernels.

### Memory Profiling

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
) as prof:
    # Training step
    outputs = model(inputs)
    loss = outputs.loss
    loss.backward()

# Print memory events
print(prof.key_averages().table(
    sort_by="self_cuda_memory_usage",
    row_limit=10,
))
```

The memory timeline in TensorBoard shows:
- **Allocations:** When and how much memory was allocated.
- **Peak usage:** The maximum memory consumed.
- **Fragmentation:** Memory that is allocated but not usable due to non-contiguous free blocks.

This is invaluable for debugging out-of-memory errors — you can see exactly which operation pushed memory over the limit.

---

## 14.5 NVIDIA Nsight Systems and Nsight Compute

While the PyTorch Profiler provides operator-level information, NVIDIA's profiling tools go deeper into GPU hardware behavior.

### Nsight Systems: System-Wide Timeline

**Nsight Systems** provides a timeline view of all CPU and GPU activity, including:

- CUDA API calls (kernel launches, memory copies, synchronization)
- GPU kernel execution
- NVTX ranges (user annotations)
- CPU thread activity
- NCCL communication

```bash
# Profile a training run
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=training_profile \
    --cuda-memory-usage=true \
    python train.py

# Open the report in Nsight Systems GUI
nsys-ui training_profile.nsys-rep
```

**Key analysis capabilities:**

1. **GPU idle time:** Zoom into the timeline to see gaps between kernels. Long gaps indicate CPU-side bottlenecks.

2. **Kernel overlap:** Check if compute kernels overlap with communication (NCCL) kernels on different CUDA streams. In well-optimized distributed training, gradient all-reduce should overlap with backward computation.

3. **CUDA stream analysis:** Multiple streams allow concurrent execution. Look for serialization points where streams synchronize unnecessarily.

4. **CPU-GPU synchronization:** Identify `cudaStreamSynchronize` and `cudaDeviceSynchronize` calls that force the CPU to wait for the GPU.

```python
# Add NVTX markers for better Nsight Systems visualization
import torch.cuda.nvtx as nvtx

for step, batch in enumerate(dataloader):
    nvtx.range_push(f"Step {step}")

    nvtx.range_push("Forward")
    outputs = model(batch)
    nvtx.range_pop()

    nvtx.range_push("Backward")
    loss = outputs.loss
    loss.backward()
    nvtx.range_pop()

    nvtx.range_push("Optimizer")
    optimizer.step()
    optimizer.zero_grad()
    nvtx.range_pop()

    nvtx.range_pop()  # End of step
```

### Nsight Compute: Kernel-Level Analysis

**Nsight Compute** profiles individual CUDA kernels in exhaustive detail:

```bash
# Profile specific kernels
ncu --set full \
    --kernel-name "volta_fp16_s884gemm" \
    --launch-skip 100 \
    --launch-count 5 \
    python train.py
```

**Key metrics from Nsight Compute:**

1. **SM Utilization:** What fraction of SMs are active during the kernel. Low utilization (<50%) suggests insufficient parallelism (too few thread blocks).

2. **Warp Occupancy:** How many warps are active versus the maximum on each SM. Low occupancy means the SM cannot hide memory latency through warp switching. Common causes: too many registers per thread, too much shared memory per block.

3. **Memory Throughput:** How much of the HBM bandwidth is utilized. Compare to peak bandwidth.
   - L1 hit rate: High hit rates reduce HBM traffic.
   - L2 hit rate: Similar — high rates indicate good data locality.

4. **Compute Throughput:** What fraction of Tensor Core or CUDA Core peak is achieved. For GEMM kernels, this should be >60%.

5. **Memory access patterns:**
   - **Coalesced accesses:** Adjacent threads access adjacent memory locations — achieves full bandwidth.
   - **Uncoalesced accesses:** Threads access scattered memory — wastes bandwidth (up to 32x penalty).
   - **Bank conflicts:** Multiple threads in a warp access the same shared memory bank — serializes accesses.

6. **Instruction mix:** Breakdown of operations (floating-point, integer, memory, control flow). High control-flow percentages indicate divergent branches.

### When to Use Each Tool

| Scenario | Tool |
|---|---|
| "Why is my training step slow?" | PyTorch Profiler or Nsight Systems |
| "Where is the GPU idle?" | Nsight Systems timeline |
| "Why is this kernel slow?" | Nsight Compute |
| "Is communication overlapping compute?" | Nsight Systems |
| "Which operator uses the most memory?" | PyTorch Profiler (memory mode) |
| "What is my SM/Tensor Core utilization?" | Nsight Compute |

---

## 14.6 Custom Triton Kernels

**Triton** (Tillet et al., 2019) is an open-source language and compiler from OpenAI that makes GPU programming accessible to Python developers. It bridges the gap between high-level PyTorch operations and low-level CUDA programming.

### Why Triton?

1. **Kernel fusion:** Combine multiple operations into a single GPU kernel, eliminating intermediate memory reads/writes.
2. **Custom operations:** Implement operations not available in PyTorch, or customize existing ones.
3. **Approachable:** Python syntax with automatic memory management, thread scheduling, and optimization — much simpler than CUDA.

### Triton Programming Model

In Triton, you write **programs** that operate on **blocks** of data (tiles). The compiler handles:
- Thread scheduling within blocks
- Memory coalescing
- Shared memory management
- Vectorized loads/stores

```python
import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(
    x_ptr,       # Pointer to first input tensor
    y_ptr,       # Pointer to second input tensor
    output_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Compile-time constant
):
    # Each program instance processes BLOCK_SIZE elements
    pid = tl.program_id(axis=0)  # Which block am I?

    # Compute the range of elements this block will process
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask for elements within bounds
    mask = offsets < n_elements

    # Load data from HBM into SRAM
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute (in SRAM)
    output = x + y

    # Store result back to HBM
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()

    # Launch kernel with enough blocks to cover all elements
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output
```

### Example: Fused Softmax Kernel

Softmax is memory-bound in standard implementations because it requires three passes over the data (find max, compute exp and sum, normalize). A fused kernel does it in a single pass through HBM:

```python
@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per row
    row_idx = tl.program_id(0)

    # Pointer to the start of the row
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # Load the entire row into SRAM
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))

    # Compute softmax in SRAM (no HBM round-trips)
    # Step 1: Subtract max for numerical stability
    row_max = tl.max(row, axis=0)
    row = row - row_max

    # Step 2: Exponentiate
    numerator = tl.exp(row)

    # Step 3: Normalize
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # Store result
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)

    # Block size must be a power of 2 >= n_cols
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # One program per row
    softmax_kernel[(n_rows,)](
        output, x,
        x.stride(0), output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output

# Benchmark
x = torch.randn(8192, 4096, device='cuda', dtype=torch.float32)

# PyTorch softmax
torch_out = torch.softmax(x, dim=-1)

# Triton softmax
triton_out = softmax(x)

# Verify correctness
assert torch.allclose(torch_out, triton_out, atol=1e-5)
```

**Why this is faster:** The standard PyTorch softmax requires reading the input from HBM three times (max, exp+sum, normalize) and writing intermediate results. The fused Triton kernel reads from HBM once and writes once, reducing HBM traffic by ~3x.

### Comparison to CUDA

| Aspect | Triton | CUDA |
|---|---|---|
| Language | Python (with decorators) | C++ |
| Thread management | Automatic (block-level) | Manual (thread-level) |
| Memory management | Automatic tiling | Manual shared memory |
| Portability | Compiles to PTX | NVIDIA-specific |
| Learning curve | Moderate | Steep |
| Performance ceiling | ~80-95% of expert CUDA | 100% (reference) |
| Development time | Hours | Days to weeks |

### Auto-tuning

Triton supports automatic hyperparameter tuning:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Tiled matrix multiplication
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute block pointers
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over K dimension in tiles
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptr + (offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak),
                     mask=(offs_m[:, None] < M) & ((k + offs_k[None, :]) < K), other=0.0)
        b = tl.load(b_ptr + ((k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn),
                     mask=((k + offs_k[:, None]) < K) & (offs_n[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)

    c = accumulator.to(tl.float16)
    tl.store(c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn),
             c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

---

## 14.7 CUDA Programming Fundamentals

While Triton abstracts away much of GPU programming, understanding CUDA fundamentals is essential for interpreting profiler output, reading GPU kernel code, and understanding performance characteristics.

### Thread Hierarchy

CUDA organizes parallel execution in a hierarchy:

```
Grid (all thread blocks for one kernel launch)
  └── Block (up to 1024 threads)
        └── Warp (32 threads — the execution unit)
              └── Thread (individual unit of execution)
```

**Threads:** The smallest unit. Each thread has its own program counter, registers, and local memory.

**Warps:** Groups of 32 threads that execute in lockstep (SIMT — Single Instruction, Multiple Threads). All threads in a warp execute the same instruction simultaneously. **Warp divergence** occurs when threads in a warp take different branches — the warp must execute both paths sequentially, wasting half the throughput.

**Blocks (Thread Blocks):** Groups of warps (up to 1024 threads). Threads within a block can communicate via shared memory and synchronize with `__syncthreads()`. Different blocks execute independently and cannot communicate directly.

**Grid:** All the blocks launched for a kernel. The grid can be 1D, 2D, or 3D.

### A Simple CUDA Kernel

```cpp
// Vector addition kernel
__global__ void vector_add(float* a, float* b, float* c, int n) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Launch configuration
int n = 1000000;
int threads_per_block = 256;
int blocks = (n + threads_per_block - 1) / threads_per_block;
vector_add<<<blocks, threads_per_block>>>(a, b, c, n);
```

### Memory Model

```
Per-Thread:
    Registers          — fastest, limited (~255 per thread)
    Local Memory       — spills to HBM (slow)

Per-Block:
    Shared Memory      — fast, 228 KB per SM (H100)
    __shared__ float s_data[256];

Per-Grid:
    Global Memory (HBM) — 80 GB, slowest
    Constant Memory     — cached, read-only, 64 KB
    Texture Memory      — cached, spatially optimized
```

### Memory Coalescing

**Coalesced access:** When consecutive threads access consecutive memory addresses, the hardware combines multiple accesses into a single wide transaction (128 bytes). This achieves full memory bandwidth.

```cpp
// COALESCED: Thread i accesses element i (contiguous)
float val = global_array[threadIdx.x];  // Good!

// UNCOALESCED: Thread i accesses element i*stride (strided)
float val = global_array[threadIdx.x * stride];  // Bad for stride > 1!

// WORST: Random access
float val = global_array[random_index[threadIdx.x]];  // Very bad!
```

Bandwidth penalty for uncoalesced access: up to 32x (each thread triggers a separate 128-byte transaction for a single 4-byte float).

### Shared Memory and Bank Conflicts

Shared memory is divided into 32 **banks** (matching the warp size). Consecutive 4-byte words are assigned to consecutive banks. When two threads in a warp access the same bank, a **bank conflict** occurs and accesses are serialized.

```cpp
__shared__ float smem[32];

// No bank conflict: thread i accesses bank i
float val = smem[threadIdx.x];           // 32 banks accessed in parallel

// Bank conflict: stride of 2 means threads 0 and 16 access bank 0
float val = smem[threadIdx.x * 2];       // 2-way bank conflict (2x slower)

// Worst case: all threads access same bank
float val = smem[0];                     // 32-way bank conflict (32x slower)
// (Broadcast exception: if ALL threads access the SAME address, it's fast)
```

### Synchronization

```cpp
// Within a block
__syncthreads();  // All threads in the block must reach this point before any proceed

// Across blocks (require multiple kernel launches or atomic operations)
atomicAdd(&global_counter, 1);  // Atomic operation visible to all blocks
```

### Writing a CUDA Kernel for PyTorch

```cpp
// softmax_cuda_kernel.cu
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_cols
) {
    int row = blockIdx.x;
    const float* input_row = input + row * n_cols;
    float* output_row = output + row * n_cols;

    // Step 1: Find max (reduction within block)
    extern __shared__ float shared[];
    float thread_max = -INFINITY;
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
        thread_max = fmaxf(thread_max, input_row[i]);
    }
    shared[threadIdx.x] = thread_max;
    __syncthreads();

    // Warp-level reduction for max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x],
                                         shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float row_max = shared[0];

    // Step 2: Compute exp and sum
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
        float val = expf(input_row[i] - row_max);
        output_row[i] = val;
        thread_sum += val;
    }
    shared[threadIdx.x] = thread_sum;
    __syncthreads();

    // Reduction for sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float row_sum = shared[0];

    // Step 3: Normalize
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
        output_row[i] /= row_sum;
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int n_rows = input.size(0);
    int n_cols = input.size(1);
    int threads = 256;

    softmax_kernel<<<n_rows, threads, threads * sizeof(float)>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n_cols
    );
    return output;
}

// Register as PyTorch extension
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax", &softmax_cuda, "Custom CUDA softmax");
}
```

Compare this to the Triton version in Section 14.6 — the CUDA version requires explicit thread management, shared memory allocation, synchronization barriers, and reduction patterns. The Triton version accomplishes the same with ~20 lines of Python.

---

## 14.8 End-to-End Performance Optimization Case Study

Let us walk through a realistic performance optimization scenario: profiling and optimizing the training of a 7B language model on 8 A100 GPUs.

### Step 1: Baseline Measurement

```python
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Baseline configuration
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float32,  # Start with FP32 as baseline
).cuda()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Synthetic data
batch_size = 1  # Small due to FP32 memory
seq_length = 2048
input_ids = torch.randint(0, 32000, (batch_size, seq_length)).cuda()

# Measure baseline
torch.cuda.synchronize()
start = time.perf_counter()

for step in range(10):
    outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

torch.cuda.synchronize()
elapsed = time.perf_counter() - start

tokens_per_second = batch_size * seq_length * 10 / elapsed
print(f"Baseline: {tokens_per_second:.0f} tokens/sec, {elapsed/10:.2f} sec/step")
```

**Baseline results (single A100):**
- Batch size: 1 (FP32 limits us to tiny batches)
- Throughput: ~1,200 tokens/sec
- Step time: ~1.7 sec
- Memory: 76 GB / 80 GB (nearly full)
- MFU: ~2.7% (terrible)

### Step 2: Profile and Identify Bottlenecks

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
) as prof:
    outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss
    loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
```

**Profile findings:**
1. Matrix multiplications consume 45% of GPU time — but in FP32, not using Tensor Cores.
2. Attention computation is 25% of time — not using Flash Attention.
3. Batch size of 1 means low SM utilization (~30%).
4. No data loading overlap (not yet using DataLoader).

### Step 3: Apply Mixed Precision (BF16)

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,  # BF16 parameters
).cuda()

# BF16 training with autocast
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss
loss.backward()
```

**Results after BF16:**
- Batch size: 4 (memory reduced ~50%)
- Throughput: ~9,600 tokens/sec (8x improvement)
- MFU: ~21%
- Memory: 42 GB / 80 GB

### Step 4: Enable Flash Attention

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # Use Flash Attention
).cuda()
```

**Results after Flash Attention:**
- Batch size: 8 (attention memory reduced from O(N^2) to O(N))
- Throughput: ~22,000 tokens/sec
- MFU: ~24%
- Memory: 55 GB / 80 GB

### Step 5: Apply Gradient Checkpointing

```python
model.gradient_checkpointing_enable()
# This recomputes activations during backward, trading compute for memory
```

**Results after gradient checkpointing:**
- Batch size: 16 (activation memory reduced by ~60%)
- Throughput: ~28,000 tokens/sec (throughput increases despite recomputation because larger batch provides better GPU utilization)
- MFU: ~31% (note: recomputed FLOPs are not counted in MFU, so MFU appears lower than actual hardware utilization)
- Memory: 68 GB / 80 GB

### Step 6: Scale to 8 GPUs with FSDP

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=mp_policy,
    auto_wrap_policy=transformer_auto_wrap_policy(
        transformer_layer_cls={LlamaDecoderLayer}
    ),
)
```

**Results after FSDP on 8 GPUs:**
- Per-GPU batch size: 16 (same as single GPU, but with sharded model)
- Global batch size: 128
- Throughput: ~180,000 tokens/sec
- Scaling efficiency: 180K / (28K * 8) = 80% (20% lost to communication)
- MFU: ~25%

### Step 7: Optimize Data Loading

```python
from torch.utils.data import DataLoader, DistributedSampler

dataloader = DataLoader(
    dataset,
    batch_size=16,
    sampler=DistributedSampler(dataset),
    num_workers=8,          # Parallel data loading
    pin_memory=True,        # Faster CPU→GPU transfer
    prefetch_factor=4,      # Prefetch 4 batches per worker
    persistent_workers=True, # Keep workers alive between epochs
)
```

**Results after data loading optimization:**
- Throughput: ~195,000 tokens/sec (8% improvement — data loading was a small bottleneck)
- MFU: ~27%

### Step 8: torch.compile

```python
model = torch.compile(model, mode="reduce-overhead")
```

`torch.compile` fuses small operations (layer norm, GeLU, residual adds) into single kernels, reducing kernel launch overhead and HBM traffic.

**Results after torch.compile:**
- Throughput: ~220,000 tokens/sec (13% improvement from kernel fusion)
- MFU: ~30%

### Final Results Summary

| Optimization | Tokens/sec | Speedup | Cumulative | MFU |
|---|---|---|---|---|
| Baseline (FP32, BS=1, 1 GPU) | 1,200 | 1.0x | 1.0x | 2.7% |
| + BF16 | 9,600 | 8.0x | 8.0x | 21% |
| + Flash Attention | 22,000 | 2.3x | 18.3x | 24% |
| + Gradient Checkpointing | 28,000 | 1.3x | 23.3x | 31% |
| + FSDP (8 GPUs) | 180,000 | 6.4x | 150x | 25% |
| + Data Loading | 195,000 | 1.1x | 162x | 27% |
| + torch.compile | 220,000 | 1.1x | 183x | 30% |

**183x total improvement** from a naive baseline to an optimized distributed training setup. The individual optimizations compound: BF16 enables larger batches, Flash Attention frees memory for even larger batches, gradient checkpointing frees more memory, FSDP distributes across GPUs, and torch.compile reduces overhead.

### Optimization Priority Order

Based on this case study and general experience, the priority order for optimization is:

1. **Mixed precision (BF16/FP16):** Always the first step. Largest single impact.
2. **Flash Attention:** Significant for any attention-based model.
3. **Batch size maximization:** Larger batches = better GPU utilization. Use gradient checkpointing and FSDP/ZeRO to enable larger batches.
4. **Distributed training:** Scale to multiple GPUs with FSDP or DDP.
5. **Data loading:** Ensure the pipeline does not starve the GPU.
6. **torch.compile / kernel fusion:** Reduces overhead from small operations.
7. **Custom Triton kernels:** For specific bottleneck operations.
8. **Communication optimization:** Overlap, compression, topology-aware placement.

---

## Exercises

1. **Arithmetic Intensity:** Calculate the arithmetic intensity of the following operations in BF16:
   (a) Matrix multiplication: $C_{4096 \times 4096} = A_{4096 \times 4096} \times B_{4096 \times 4096}$
   (b) Element-wise GELU activation on a tensor of shape (32, 2048, 4096)
   (c) Layer normalization on a tensor of shape (32, 2048, 4096)
   Classify each as compute-bound or memory-bound on H100.

2. **MFU Calculation:** You are training a 13B parameter model with batch size 64 and sequence length 4096. Each training step takes 4.2 seconds on 8 A100 GPUs.
   (a) Calculate the total model FLOPs per step.
   (b) Calculate the achieved FLOP/s.
   (c) Calculate the MFU (A100 peak BF16 = 312 TFLOPS).
   (d) Is this good? What might be limiting MFU?

3. **Profiling Practice:** Write a profiling script for a 3-layer transformer model that:
   (a) Uses `record_function` to annotate forward, backward, and optimizer phases.
   (b) Captures 5 profiling steps with 2 warmup steps.
   (c) Exports results to TensorBoard.
   (d) Prints the top 10 operations by CUDA time.

4. **Triton Kernel:** Write a fused kernel in Triton that computes $\text{output} = \text{GELU}(x \cdot w + b)$ where $x$ is the input, $w$ is a weight vector (element-wise multiply), and $b$ is a bias vector. Compare its performance to the equivalent three separate PyTorch operations.

5. **Memory Analysis:** For an A100 80GB GPU running a 7B parameter model in BF16:
   (a) Calculate memory for model parameters, gradients, and AdamW optimizer states.
   (b) Estimate activation memory for batch size 4, sequence length 2048 (assume 2 bytes per element, stored at each layer boundary).
   (c) What is the maximum batch size that fits in memory without gradient checkpointing?
   (d) What is the maximum batch size with gradient checkpointing (assume 2x activation memory reduction)?

6. **Roofline Analysis:** Draw a roofline diagram for A100 (peak BF16 = 312 TFLOPS, bandwidth = 2 TB/s). Mark the ridge point. Plot the following operations and determine if each is compute-bound or memory-bound:
   (a) GEMM with M=N=K=8192
   (b) Softmax on 8192 elements
   (c) Batch normalization on (64, 256, 32, 32) tensor
   (d) Attention (Q, K, V each 32x2048x128, no Flash Attention)

7. **End-to-End Optimization:** You observe the following profile for a training step:
   - Data loading: 50ms
   - Forward pass: 200ms
   - Backward pass: 400ms
   - All-reduce: 100ms (not overlapped)
   - Optimizer step: 30ms
   - Total: 780ms
   (a) What is the GPU utilization?
   (b) Propose three optimizations and estimate the speedup from each.
   (c) What would be the total step time after all optimizations?

---

## References

Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., ... & Fiedel, N. (2022). PaLM: Scaling language modeling with Pathways. *Journal of Machine Learning Research*, 24(240), 1-113.

NVIDIA (2023). *CUDA C++ Programming Guide*. https://docs.nvidia.com/cuda/cuda-c-programming-guide/.

NVIDIA (2023). *NVIDIA A100 Tensor Core GPU Architecture*. White paper.

NVIDIA (2023). *NVIDIA H100 Tensor Core GPU Architecture*. White paper.

PyTorch (2023). *PyTorch Profiler*. https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html.

Smith, S., Patwary, M., Norick, B., LeGresley, P., Rajbhandari, S., Casper, J., ... & Catanzaro, B. (2022). Using DeepSpeed and Megatron to train Megatron-Turing NLG 530B, a large-scale generative language model. *arXiv preprint arXiv:2201.11990*.

Tillet, P., Kung, H. T., & Cox, D. (2019). Triton: An intermediate language and compiler for tiled neural network computations. *Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages*, 10-19.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M. A., Lacroix, T., ... & Lample, G. (2023). LLaMA: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*.

Williams, S., Waterman, A., & Patterson, D. (2009). Roofline: An insightful visual performance model for multicore architectures. *Communications of the ACM*, 52(4), 65-76.
