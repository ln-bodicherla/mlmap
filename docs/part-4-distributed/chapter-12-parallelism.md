# Chapter 12: Parallelism Strategies

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain why distributed training is necessary and identify the fundamental bottlenecks in single-GPU training.
2. Implement Data Parallelism using PyTorch's DistributedDataParallel (DDP) and mathematically verify gradient correctness under data-parallel aggregation.
3. Use gradient accumulation to simulate large batch sizes on memory-constrained hardware.
4. Understand Fully Sharded Data Parallel (FSDP) and its memory savings over vanilla data parallelism.
5. Compare DeepSpeed ZeRO Stages 1, 2, and 3, and analyze memory consumption at each stage.
6. Describe Tensor Parallelism, Pipeline Parallelism, Sequence Parallelism, and Expert Parallelism, and know when to apply each strategy.
7. Design a 3D parallelism configuration for large-scale model training on a multi-node cluster.

---

## 12.1 Why Distributed Training Is Necessary

The modern era of deep learning is defined by scale. Between 2018 and 2024, the parameter count of frontier language models grew from approximately 340 million (BERT) to over 1.8 trillion (GPT-4 class models), representing a roughly 10x increase per year in some periods (Sevilla et al., 2022). This exponential growth has far outpaced the improvements in single-GPU memory and compute.

### The Single-GPU Wall

Consider the memory required to train a model with $\Psi$ parameters in mixed-precision (FP16/BF16 forward and backward, FP32 optimizer states). The memory footprint breaks down as follows:

- **Model parameters (FP16):** $2\Psi$ bytes
- **Gradients (FP16):** $2\Psi$ bytes
- **Optimizer states (Adam, FP32):** $12\Psi$ bytes (FP32 master weights: $4\Psi$, first moment: $4\Psi$, second moment: $4\Psi$)
- **Activations:** variable, often dominant for long sequences

**Total (excluding activations):** $16\Psi$ bytes

For a 7-billion-parameter model: $16 \times 7 \times 10^9 = 112$ GB — already exceeding the 80 GB of an NVIDIA A100 or H100 GPU, before we even account for activations.

For a 70B parameter model, the training memory requirement is approximately $70 \times 16 = 1,120$ GB, or **1.1 TB** — requiring at minimum 14 A100 80GB GPUs just for the static memory, ignoring activations entirely.

### The Growth of Model Sizes

To put the scale problem in perspective, consider the trajectory of landmark models:

| Model | Year | Parameters | Training Compute (FLOP) | GPUs Used |
|---|---|---|---|---|
| BERT-Large | 2018 | 340M | ~$10^{18}$ | 16 TPUv3 |
| GPT-2 | 2019 | 1.5B | ~$10^{19}$ | 256 V100 |
| GPT-3 | 2020 | 175B | $3.14 \times 10^{23}$ | 10,000 V100 |
| PaLM | 2022 | 540B | $2.5 \times 10^{24}$ | 6,144 TPUv4 |
| LLaMA-2 70B | 2023 | 70B | $1.7 \times 10^{24}$ | 2,048 A100 |
| LLaMA-3 405B | 2024 | 405B | ~$4 \times 10^{25}$ | 16,384 H100 |

Each generation requires roughly 10x more compute, while GPU compute capacity doubles approximately every 2 years (following a slower-than-Moore's-Law trajectory). This gap necessitates distributing training across increasingly large GPU clusters.

### Communication Costs

Distributing training across multiple GPUs introduces communication overhead. The key metrics are:

- **Bandwidth:** NVLink provides 600 GB/s (A100) to 900 GB/s (H100) between GPUs within a node. InfiniBand provides 200-400 Gb/s between nodes.
- **Latency:** Intra-node communication is on the order of microseconds; inter-node communication adds tens of microseconds.
- **Collective operations:** All-reduce, all-gather, and reduce-scatter are the fundamental primitives. Their cost depends on the message size and the number of participants.

The choice of parallelism strategy is fundamentally about minimizing the ratio of communication time to computation time while distributing memory evenly across devices.

### The Fundamental Trade-off

Every distributed training strategy navigates the same fundamental trade-off:

$$\text{Total time} = \text{Compute time} + \text{Communication time} + \text{Idle time}$$

The goal is to minimize the total time by:
1. **Maximizing compute utilization:** Keeping all GPUs busy with useful work.
2. **Overlapping communication with computation:** Using asynchronous communication primitives so that data transfer happens while GPUs are computing.
3. **Minimizing idle time:** Reducing pipeline bubbles and synchronization barriers.
4. **Balancing memory:** Ensuring no single GPU becomes a memory bottleneck that forces smaller batch sizes or model configurations.

### The Parallelism Taxonomy

| Strategy | What is partitioned | Communication pattern | Primary benefit |
|---|---|---|---|
| Data Parallelism | Data (mini-batches) | All-reduce on gradients | Linear throughput scaling |
| Tensor Parallelism | Weight matrices (intra-layer) | All-reduce / all-gather per layer | Reduces per-GPU memory |
| Pipeline Parallelism | Model layers (inter-layer) | Point-to-point between stages | Reduces per-GPU memory |
| Sequence Parallelism | Activations along sequence dim | All-gather / reduce-scatter | Reduces activation memory |
| Expert Parallelism | Expert subnetworks | All-to-all routing | Scales MoE models |

In practice, large-scale training combines multiple strategies — an approach known as **3D parallelism** or **N-dimensional parallelism** (Narayanan et al., 2021).

---

## 12.2 Data Parallelism and PyTorch DDP

Data parallelism is the simplest and most widely used distributed training strategy. The idea is straightforward: replicate the model on every GPU, partition the training data across GPUs, and synchronize gradients after each backward pass.

### Mathematical Foundation

Let $f(\theta; x)$ denote the loss function for a model with parameters $\theta$ on input $x$. For a mini-batch $\mathcal{B} = \{x_1, x_2, \ldots, x_B\}$, the gradient is:

$$\nabla_\theta \mathcal{L} = \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta f(\theta; x_i)$$

Now partition $\mathcal{B}$ into $N$ disjoint subsets $\mathcal{B}_1, \mathcal{B}_2, \ldots, \mathcal{B}_N$ where $|\mathcal{B}_k| = B/N$ for each GPU $k$. Each GPU computes its local gradient:

$$g_k = \frac{1}{B/N} \sum_{x_i \in \mathcal{B}_k} \nabla_\theta f(\theta; x_i)$$

The average of local gradients recovers the true mini-batch gradient:

$$\frac{1}{N} \sum_{k=1}^{N} g_k = \frac{1}{N} \sum_{k=1}^{N} \frac{N}{B} \sum_{x_i \in \mathcal{B}_k} \nabla_\theta f(\theta; x_i) = \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta f(\theta; x_i) = \nabla_\theta \mathcal{L}$$

**Proof of correctness:** The last equality holds because $\{\mathcal{B}_k\}_{k=1}^N$ forms a partition of $\mathcal{B}$, so the double sum over all subsets covers every element of the mini-batch exactly once. The average of local averages equals the global average. $\square$

This means that performing an **all-reduce** (specifically, a mean-reduce) on the local gradients yields the mathematically identical gradient to computing on the full batch on a single device.

### Scaling Properties of Data Parallelism

**Linear speedup (ideal case):** With $N$ GPUs, the per-GPU computation is $1/N$ of the total. If communication cost is zero, we get perfect $N\times$ speedup. In practice, the speedup is:

$$\text{Speedup} = \frac{N}{1 + \frac{t_{\text{comm}}}{t_{\text{comp}}}}$$

where $t_{\text{comm}}$ is the communication time and $t_{\text{comp}}$ is the per-GPU computation time.

**Large-batch training considerations:** Data parallelism increases the effective batch size. However, very large batch sizes can degrade generalization. The **linear scaling rule** (Goyal et al., 2017) suggests scaling the learning rate proportionally: if you multiply the batch size by $k$, multiply the learning rate by $k$. Additionally, a **warmup period** of gradually increasing the learning rate helps stabilize training at large batch sizes.

The LARS (Layer-wise Adaptive Rate Scaling) and LAMB (Layer-wise Adaptive Moments) optimizers were specifically designed for large-batch training, applying per-layer learning rate scaling based on the ratio of weight norm to gradient norm (You et al., 2020).

### All-Reduce Algorithms

The all-reduce operation is the communication backbone of data parallelism. Given $N$ GPUs each holding a vector $g_k$, all-reduce computes $\bar{g} = \frac{1}{N}\sum_k g_k$ and distributes the result to all GPUs.

**Ring All-Reduce:** Each GPU sends a chunk of its data to the next GPU in a logical ring. After $2(N-1)$ steps, the operation completes. The total data transferred per GPU is $2 \cdot \frac{(N-1)}{N} \cdot M$ where $M$ is the message size. This is bandwidth-optimal — the per-GPU transfer volume is independent of $N$ for large $M$.

**Tree All-Reduce:** Organizes GPUs in a binary tree. A reduce phase propagates partial sums up the tree, and a broadcast phase distributes the result back down. Total latency is $O(\log N)$ but bandwidth is not optimal.

**NCCL (NVIDIA Collective Communications Library)** automatically selects the best algorithm based on message size, GPU topology, and interconnect (Jeaugey, 2017). For small messages, tree-based algorithms minimize latency; for large messages, ring-based algorithms maximize bandwidth.

### PyTorch DDP Implementation

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def setup(rank, world_size):
    """Initialize the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize process group with NCCL backend (optimized for GPUs)
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, dataset, model_cls, epochs=10):
    setup(rank, world_size)

    # Create model and move to GPU
    model = model_cls().to(rank)

    # Wrap with DDP — this handles gradient synchronization automatically
    ddp_model = DDP(model, device_ids=[rank])

    # DistributedSampler ensures each GPU gets a unique subset of data
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # IMPORTANT: set epoch so sampler shuffles differently each epoch
        sampler.set_epoch(epoch)

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(rank, non_blocking=True)
            targets = targets.to(rank, non_blocking=True)

            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()   # Gradient all-reduce happens here automatically
            optimizer.step()

    cleanup()

# Launch with torchrun (recommended):
# torchrun --nproc_per_node=4 train_script.py

# Or launch programmatically:
if __name__ == "__main__":
    import torch.multiprocessing as mp
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, dataset, MyModel), nprocs=world_size)
```

**Key implementation details:**

1. **Process groups:** DDP uses NCCL process groups to manage GPU-to-GPU communication. All GPUs in a group participate in collective operations. You can create sub-groups for hierarchical communication patterns.

2. **Gradient bucketing:** DDP does not wait for the entire backward pass to complete. Instead, it groups gradients into buckets (default 25 MB) and overlaps the all-reduce of completed buckets with the backward computation of remaining layers. This overlap is critical for hiding communication latency. The bucket size can be tuned via the `bucket_cap_mb` parameter — larger buckets improve bandwidth utilization but reduce overlap opportunity.

3. **DistributedSampler:** Ensures each GPU processes a unique partition of the data. Without it, all GPUs would train on identical data, negating the purpose of data parallelism. The sampler also handles epoch-wise shuffling via `set_epoch()`.

4. **Synchronization:** By default, DDP synchronizes gradients on every backward pass. The `no_sync()` context manager can defer synchronization for gradient accumulation.

5. **Broadcasting at startup:** DDP broadcasts model parameters from rank 0 to all other ranks during initialization, ensuring all replicas start with identical weights.

6. **Static graph optimization:** For models with a fixed computation graph (no dynamic control flow), setting `static_graph=True` enables additional optimizations like communication scheduling.

### DDP Gradient Bucketing In Detail

The gradient bucketing mechanism deserves deeper examination because it is the key to DDP's performance. The process works as follows:

```
Layer 1 (first in forward, last in backward):
    backward computation → gradients added to bucket A

Layer 2:
    backward computation → gradients added to bucket A
    [Bucket A is full → launch all-reduce for bucket A]

Layer 3:
    backward computation → gradients added to bucket B
    [Meanwhile, bucket A all-reduce is executing asynchronously]

Layer 4:
    backward computation → gradients added to bucket B
    [Bucket B is full → launch all-reduce for bucket B]
    [Bucket A all-reduce may have completed by now]
```

This pipelining means that, for models with many layers, the all-reduce for early buckets completes by the time the backward pass finishes — effectively hiding most of the communication latency. The degree of overlap depends on:

- The ratio of computation time per layer to communication time per bucket.
- The number and size of buckets.
- The network bandwidth and latency.

### Multi-Node DDP with torchrun

For multi-node training, `torchrun` handles process coordination across machines:

```bash
# On node 0 (master):
torchrun \
    --nnodes=4 \
    --nproc_per_node=8 \
    --node_rank=0 \
    --master_addr=10.0.0.1 \
    --master_port=29500 \
    train.py

# On node 1:
torchrun \
    --nnodes=4 \
    --nproc_per_node=8 \
    --node_rank=1 \
    --master_addr=10.0.0.1 \
    --master_port=29500 \
    train.py
```

The master address and port are used for initial rendezvous — all processes connect to this endpoint to discover each other. After initialization, NCCL handles direct GPU-to-GPU communication.

---

## 12.3 Gradient Accumulation

When GPU memory is insufficient to hold the desired batch size, or when training requires very large effective batch sizes (common in LLM training), **gradient accumulation** provides a solution without additional GPUs.

### Core Idea

Instead of performing an optimizer step after every forward-backward pass, accumulate gradients over $K$ micro-batches before stepping:

$$g_{\text{accumulated}} = \sum_{k=1}^{K} \nabla_\theta \mathcal{L}_k$$

The effective batch size becomes:

$$B_{\text{effective}} = B_{\text{micro}} \times K \times N_{\text{GPUs}}$$

where $B_{\text{micro}}$ is the per-GPU micro-batch size, $K$ is the accumulation steps, and $N_{\text{GPUs}}$ is the number of GPUs.

### Implementation

```python
accumulation_steps = 4
optimizer.zero_grad()

for step, (inputs, targets) in enumerate(dataloader):
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Use no_sync() to skip gradient all-reduce on intermediate steps
    # (only relevant for DDP)
    context = ddp_model.no_sync if (step + 1) % accumulation_steps != 0 else nullcontext

    with context():
        outputs = ddp_model(inputs)
        # Normalize loss by accumulation steps to maintain correct gradient scale
        loss = loss_fn(outputs, targets) / accumulation_steps
        loss.backward()

    if (step + 1) % accumulation_steps == 0:
        # Optional: gradient clipping
        torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
```

### Gradient Normalization

The division by `accumulation_steps` in the loss computation is essential. Without it, the accumulated gradient would be $K$ times larger than the true batch gradient. Dividing the loss (and therefore the gradient) by $K$ ensures:

$$\frac{1}{K} \sum_{k=1}^{K} \nabla_\theta \mathcal{L}_k = \nabla_\theta \left[\frac{1}{K} \sum_{k=1}^{K} \mathcal{L}_k\right]$$

which is mathematically equivalent to computing the gradient on the full effective batch.

### When to Use Gradient Accumulation

- **Memory-constrained training:** When the model or activations do not fit with the desired batch size.
- **Large-batch training:** LLM pretraining often uses effective batch sizes of millions of tokens. For example, LLaMA-2 used a batch size of 4 million tokens (Touvron et al., 2023).
- **Combined with DDP:** Using `no_sync()` avoids redundant all-reduce operations on intermediate accumulation steps, reducing communication overhead by a factor of $K$.

### Practical Example: Calculating Effective Batch Size

Consider training LLaMA-2 7B with the following setup:
- Per-GPU micro-batch size: 2 sequences
- Sequence length: 4096 tokens
- Gradient accumulation steps: 8
- Number of GPUs: 64

$$B_{\text{effective}} = 2 \times 8 \times 64 = 1024 \text{ sequences}$$
$$\text{Tokens per update} = 1024 \times 4096 = 4,194,304 \text{ tokens} \approx 4M \text{ tokens}$$

This matches the batch sizes used in practice for large-scale pretraining. Without gradient accumulation, achieving this batch size would require either 512 GPUs (with micro-batch size 2 each) or 64 GPUs each processing micro-batches of 16 sequences — which would not fit in memory for a 7B model.

### Gradient Accumulation Pitfalls

Several subtle issues can arise with gradient accumulation:

1. **Batch normalization:** BatchNorm statistics are computed per micro-batch, not per effective batch. For large accumulation steps, consider using LayerNorm or GroupNorm instead, or synchronize BatchNorm statistics across micro-batches.

2. **Dropout consistency:** Each micro-batch uses different dropout masks, which is generally fine and provides additional stochasticity.

3. **Learning rate interaction:** The effective gradient is the same regardless of accumulation, so the learning rate should be tuned for the effective batch size, not the micro-batch size.

---

## 12.4 Fully Sharded Data Parallel (FSDP)

Standard data parallelism replicates the full model on every GPU, which is wasteful: each GPU stores an identical copy of all parameters, gradients, and optimizer states. **Fully Sharded Data Parallel (FSDP)** (Zhao et al., 2023) eliminates this redundancy by sharding everything.

### The Key Insight

At any given moment during training, a GPU only needs:
- The **full parameters** of the layer currently being computed (forward or backward).
- The **local gradients** and **local optimizer states** for its shard of the model.

FSDP exploits this observation by:
1. **Sharding** parameters, gradients, and optimizer states across all GPUs.
2. **All-gathering** the full parameters just before they are needed for a forward or backward computation.
3. **Discarding** the non-local parameters immediately after use.
4. **Reduce-scattering** gradients so each GPU receives only its shard of the gradient.

### Communication Pattern

**Forward Pass:**
```
For each FSDP unit (typically a transformer layer):
    1. All-gather: Collect full parameters from all GPUs
    2. Compute forward pass
    3. Discard non-local parameters (free memory)
```

**Backward Pass:**
```
For each FSDP unit (in reverse order):
    1. All-gather: Collect full parameters (needed for gradient computation)
    2. Compute backward pass (local gradients)
    3. Reduce-scatter: Each GPU gets its shard of the averaged gradient
    4. Discard non-local parameters
```

### Memory Savings

For a model with $\Psi$ parameters and $N$ GPUs, compare the per-GPU memory:

| Component | DDP | FSDP |
|---|---|---|
| Parameters | $2\Psi$ | $2\Psi / N$ |
| Gradients | $2\Psi$ | $2\Psi / N$ |
| Optimizer States | $12\Psi$ | $12\Psi / N$ |
| **Total** | **$16\Psi$** | **$16\Psi / N$** |

The memory savings are **linear in the number of GPUs**. With 8 GPUs, FSDP uses 1/8 the memory of DDP per GPU.

Additionally, during forward/backward computation, temporary full parameters for the active layer consume $2\Psi_{\text{layer}}$ bytes, where $\Psi_{\text{layer}}$ is the parameter count of a single FSDP unit. For a model with $L$ layers, $\Psi_{\text{layer}} \approx \Psi / L$, which is much smaller than $\Psi$.

### Sharding Strategies

FSDP offers three sharding strategies:

1. **`FULL_SHARD`**: Shard parameters, gradients, and optimizer states. Maximum memory savings, maximum communication.
2. **`SHARD_GRAD_OP`**: Shard only gradients and optimizer states; keep full parameters. Equivalent to DeepSpeed ZeRO Stage 2. Fewer all-gathers needed.
3. **`NO_SHARD`**: No sharding — equivalent to standard DDP. Useful as a baseline.

### PyTorch FSDP Code

```python
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from transformers import AutoModelForCausalLM

# Define mixed precision policy
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

# Define wrapping policy — each transformer layer becomes an FSDP unit
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

auto_wrap_policy = transformer_auto_wrap_policy(
    transformer_layer_cls={LlamaDecoderLayer}
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
)

# Wrap with FSDP
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=mp_policy,
    auto_wrap_policy=auto_wrap_policy,
    cpu_offload=CPUOffload(offload_params=False),
    device_id=torch.cuda.current_device(),
    limit_all_gathers=True,  # Prevent memory spikes from overlapping all-gathers
)

# Training loop proceeds as normal
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for batch in dataloader:
    inputs = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Hugging Face Accelerate Integration

For practitioners who prefer a simpler interface, Hugging Face Accelerate provides FSDP integration:

```python
# accelerate_config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
  fsdp_offload_params: false
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_state_dict_type: SHARDED_STATE_DICT
mixed_precision: bf16
```

Launch with:
```bash
accelerate launch --config_file accelerate_config.yaml train.py
```

---

## 12.5 DeepSpeed ZeRO — Stages 1, 2, and 3

DeepSpeed's **Zero Redundancy Optimizer (ZeRO)** (Rajbhandari et al., 2020) was the pioneering work that introduced the idea of partitioning optimizer states, gradients, and parameters across data-parallel ranks. FSDP was later inspired by and builds upon ZeRO's ideas.

### Memory Analysis

For a model with $\Psi$ parameters trained with Adam in mixed precision, the per-GPU memory for each ZeRO stage with $N$ data-parallel GPUs is:

**Baseline (no ZeRO, standard data parallelism):**

| Component | Memory (bytes) |
|---|---|
| FP16 Parameters | $2\Psi$ |
| FP16 Gradients | $2\Psi$ |
| FP32 Master Weights | $4\Psi$ |
| FP32 First Moment (Adam) | $4\Psi$ |
| FP32 Second Moment (Adam) | $4\Psi$ |
| **Total** | **$16\Psi$** |

**ZeRO Stage 1 — Partition Optimizer States:**

Each GPU stores only $1/N$ of the optimizer states (master weights + moments), but retains full parameters and gradients.

| Component | Memory (bytes) |
|---|---|
| FP16 Parameters | $2\Psi$ |
| FP16 Gradients | $2\Psi$ |
| Optimizer States | $12\Psi / N$ |
| **Total** | **$4\Psi + 12\Psi / N$** |

With $N = 64$: $\approx 4.19\Psi$ bytes per GPU (3.8x reduction).

**ZeRO Stage 2 — Partition Optimizer States + Gradients:**

Gradients are also partitioned. Each GPU only needs to retain gradients for the parameters whose optimizer states it owns.

| Component | Memory (bytes) |
|---|---|
| FP16 Parameters | $2\Psi$ |
| FP16 Gradients | $2\Psi / N$ |
| Optimizer States | $12\Psi / N$ |
| **Total** | **$2\Psi + 14\Psi / N$** |

With $N = 64$: $\approx 2.22\Psi$ bytes per GPU (7.2x reduction).

**ZeRO Stage 3 — Partition Everything:**

Parameters are also partitioned. This is equivalent to FSDP with `FULL_SHARD`.

| Component | Memory (bytes) |
|---|---|
| FP16 Parameters | $2\Psi / N$ |
| FP16 Gradients | $2\Psi / N$ |
| Optimizer States | $12\Psi / N$ |
| **Total** | **$16\Psi / N$** |

With $N = 64$: $\approx 0.25\Psi$ bytes per GPU (64x reduction).

### Communication Overhead

The memory savings come at the cost of increased communication:

- **Stage 1:** Same as DDP — one all-reduce per backward pass. Volume: $2\Psi$ bytes per GPU.
- **Stage 2:** Replace all-reduce with reduce-scatter (each GPU receives only its gradient shard). Followed by a broadcast of updated parameters. Volume: $2\Psi$ bytes per GPU — same as DDP.
- **Stage 3:** Add all-gather before each forward/backward layer computation. Volume: approximately $3\Psi$ bytes per GPU (1.5x DDP). The increase comes from all-gathering parameters in both forward and backward passes.

### DeepSpeed Configuration

```python
# ds_config_zero3.json
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "none"
        },
        "offload_optimizer": {
            "device": "none"
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
```

```python
import deepspeed

model, optimizer, _, scheduler = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config_zero3.json",
)

for batch in dataloader:
    outputs = model(batch)
    loss = outputs.loss
    model.backward(loss)
    model.step()
```

### When to Choose Each Stage

| Scenario | Recommended Stage |
|---|---|
| Model fits in GPU memory, want faster training | Stage 1 |
| Model fits but optimizer states are tight | Stage 1 |
| Model parameters fit but gradients + optimizer do not | Stage 2 |
| Model does not fit on a single GPU | Stage 3 |
| Need maximum memory efficiency | Stage 3 |

---

## 12.6 ZeRO-Infinity — CPU and NVMe Offloading

ZeRO-Infinity (Rajbhandari et al., 2021) extends ZeRO Stage 3 by offloading tensors to CPU memory or NVMe SSDs, enabling training of models that exceed the aggregate GPU memory of the cluster.

### Offloading Hierarchy

```
GPU HBM (fastest, smallest)
    ↕  PCIe / NVLink
CPU DRAM (10-100x larger, 10-30x slower)
    ↕  PCIe
NVMe SSD (100-1000x larger, 100-1000x slower)
```

### What Gets Offloaded

| Configuration | Offloaded to CPU | Offloaded to NVMe |
|---|---|---|
| Optimizer offload | Optimizer states + updates | — |
| Parameter offload | Parameters + optimizer states | — |
| NVMe offload | Overflow from CPU | Optimizer states, parameters |

### Configuration

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

For NVMe offloading:

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 5,
            "fast_init": false
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 5,
            "max_in_cpu": 1e9
        }
    }
}
```

### Performance Considerations

CPU offloading typically reduces throughput by 1.5-3x compared to pure GPU training, depending on the ratio of compute to data transfer. The key optimization is **overlap**: while the GPU computes on one layer, the next layer's parameters are being prefetched from CPU to GPU via asynchronous memory copies.

NVMe offloading adds another level of latency and is typically only viable when:
- The model is too large to fit even in aggregate CPU + GPU memory.
- The NVMe drives have high sequential bandwidth (3+ GB/s per drive, with multiple drives in RAID-0).
- The training is heavily compute-bound (so the GPU remains busy while data transfers occur).

ZeRO-Infinity demonstrated training a 1-trillion-parameter model on 512 V100 GPUs, where the model's memory footprint (16 TB for optimizer states alone) far exceeded the aggregate GPU memory (Rajbhandari et al., 2021).

---

## 12.7 Tensor Parallelism

While data parallelism and ZeRO/FSDP partition the model across the data dimension, **tensor parallelism** (also called intra-layer model parallelism) partitions individual weight matrices within a layer across GPUs. This approach was popularized by **Megatron-LM** (Shoeybi et al., 2019).

### Column-Parallel Linear Layer

Consider a linear layer $Y = XA$ where $X \in \mathbb{R}^{b \times k}$ is the input and $A \in \mathbb{R}^{k \times n}$ is the weight matrix. In column-parallel partitioning, we split $A$ column-wise across $N$ GPUs:

$$A = [A_1, A_2, \ldots, A_N]$$

where $A_i \in \mathbb{R}^{k \times (n/N)}$. Each GPU $i$ computes:

$$Y_i = X A_i \in \mathbb{R}^{b \times (n/N)}$$

The full output is $Y = [Y_1, Y_2, \ldots, Y_N]$, which can be reconstructed with an **all-gather** operation if needed, or kept distributed for the next layer.

### Row-Parallel Linear Layer

For $Y = XA$ with row-parallel partitioning, we split $A$ row-wise:

$$A = \begin{bmatrix} A_1 \\ A_2 \\ \vdots \\ A_N \end{bmatrix}$$

where $A_i \in \mathbb{R}^{(k/N) \times n}$. Each GPU holds a corresponding shard of the input $X_i \in \mathbb{R}^{b \times (k/N)}$ and computes:

$$Y_i = X_i A_i \in \mathbb{R}^{b \times n}$$

The full output requires an **all-reduce**: $Y = \sum_{i=1}^{N} Y_i$.

### Megatron-LM Transformer Partitioning

In a Transformer, each MLP block consists of two linear layers with a GeLU activation between them:

$$Y = \text{GeLU}(XA)B$$

Megatron-LM partitions this as:

1. $A$ is split **column-wise** — each GPU computes its portion of $\text{GeLU}(XA_i)$ independently (GeLU is element-wise, so no communication needed).
2. $B$ is split **row-wise** — each GPU computes $Y_i = \text{GeLU}(XA_i) B_i$.
3. An **all-reduce** combines: $Y = \sum_i Y_i$.

For the self-attention block, the Q, K, V projection matrices are split column-wise (one attention head or group of heads per GPU), and the output projection is split row-wise. This yields just **two all-reduces per transformer layer** (one for attention, one for MLP).

```
[Input X] → Column-parallel QKV → Attention (local) → Row-parallel Output → All-Reduce → LayerNorm
         → Column-parallel MLP1 → GeLU (local)     → Row-parallel MLP2  → All-Reduce → LayerNorm
```

### When to Use Tensor Parallelism

Tensor parallelism requires **high-bandwidth interconnect** between participating GPUs because it introduces communication at every layer. Therefore:

- **Intra-node (NVLink):** Tensor parallelism works well with TP degree = 2, 4, or 8 within a single node.
- **Inter-node (InfiniBand):** Generally too slow for tensor parallelism. Use data or pipeline parallelism across nodes instead.

A typical configuration for 8-GPU nodes:
- Tensor parallelism = 8 (within a node, all connected via NVLink)
- Data/pipeline parallelism across nodes

### Communication Cost Analysis

For a single transformer layer with hidden size $h$ and tensor parallelism degree $N$:

- **Column-parallel forward:** No communication needed (input $X$ is replicated on all GPUs).
- **Row-parallel forward:** One all-reduce of size $b \times s \times h \times 2$ bytes (BF16).
- **Backward pass:** One all-reduce in each direction (mirroring the forward pass).

Total communication per layer: 2 all-reduces, each of size $b \times s \times h \times 2$ bytes.

For a 70B model with $h = 8192$, $s = 4096$, $b = 1$, BF16:
$$\text{Volume per all-reduce} = 1 \times 4096 \times 8192 \times 2 = 64 \text{ MB}$$
$$\text{Per layer (2 all-reduces)} = 128 \text{ MB}$$
$$\text{Per model (80 layers)} = 10.24 \text{ GB per forward+backward pass}$$

At NVLink bandwidth of 600 GB/s: $10.24 / 600 \approx 17$ ms per training step. This is acceptable when computation takes hundreds of milliseconds.

At InfiniBand bandwidth of 50 GB/s: $10.24 / 50 \approx 205$ ms — often too slow, especially since these all-reduces happen sequentially (one per layer, blocking further computation). This is why tensor parallelism is restricted to intra-node communication.

### Tensor Parallelism Implementation Considerations

Several practical details affect tensor parallelism implementation:

1. **Attention head partitioning:** With TP degree $N$, each GPU gets $H/N$ attention heads. The number of heads must be divisible by $N$.

2. **Vocabulary parallelism:** The embedding and output projection layers (which are very large for large vocabularies) can also be partitioned column-wise across GPUs.

3. **Dropout and randomness:** Each GPU must use different random seeds for dropout within tensor-parallel regions but the same seed for data sampling.

4. **Residual connections:** The residual connection around attention and MLP blocks must add the full (all-reduced) output, not the partial output from one GPU.

5. **Position embeddings:** Must be replicated on all GPUs, as they are not partitioned.

---

## 12.8 Pipeline Parallelism

**Pipeline parallelism** partitions the model vertically by assigning different layers (or groups of layers) to different GPUs. GPU 0 might run layers 1-8, GPU 1 runs layers 9-16, and so on.

### The Bubble Problem

Naive pipeline parallelism (one micro-batch at a time) is extremely inefficient because at any given moment, only one GPU is active:

```
GPU 0: [Forward] [   idle   ] [   idle   ] [Backward] [   idle   ] [   idle   ]
GPU 1: [  idle  ] [Forward]   [   idle   ] [  idle  ] [Backward]   [   idle   ]
GPU 2: [  idle  ] [  idle  ]  [Forward]    [  idle  ] [  idle  ]   [Backward]
```

The **pipeline bubble** — the fraction of time GPUs are idle — is:

$$\text{Bubble fraction} = \frac{(P - 1)}{M + P - 1}$$

where $P$ is the number of pipeline stages and $M$ is the number of micro-batches. For this to be small, we need $M \gg P$.

### GPipe Schedule

**GPipe** (Huang et al., 2019) reduces the bubble by splitting a mini-batch into $M$ micro-batches and pipelining them:

```
GPU 0: [F1][F2][F3][F4][        ][B4][B3][B2][B1]
GPU 1: [  ][F1][F2][F3][F4][    ][B4][B3][B2][B1]
GPU 2: [  ][  ][F1][F2][F3][F4] [B4][B3][B2][B1]
GPU 3: [  ][  ][  ][F1][F2][F3] [F4][B4][B3][B2][B1]
```

GPipe runs all forward passes first, then all backward passes. It requires storing activations for all $M$ micro-batches, leading to high memory usage.

### PipeDream / 1F1B Schedule

**PipeDream** (Narayanan et al., 2019) and its variant **PipeDream-2BW** (Narayanan et al., 2021) use an interleaved **1F1B (one forward, one backward)** schedule that alternates forward and backward passes as soon as possible:

```
GPU 0: [F1][F2][F3][F4][B1][F5][B2][F6][B3][F7][B4][F8][B5][B6][B7][B8]
GPU 1: [  ][F1][F2][F3][F4][B1][F5][B2][F6][B3][F7][B4][F8][B5][B6][B7][B8]
GPU 2: [  ][  ][F1][F2][F3][F4][B1][F5][B2][F6][B3][B4]...
GPU 3: [  ][  ][  ][F1][F2][F3][F4][B1][F5][B2]...
```

The 1F1B schedule limits the number of in-flight micro-batches to $P$ (the number of stages), reducing activation memory from $O(M)$ to $O(P)$. This is a significant memory advantage.

### Interleaved Pipeline Parallelism

Narayanan et al. (2021) introduced **interleaved stages** where each GPU is assigned multiple non-contiguous groups of layers. For example, with 4 GPUs and 16 layers, instead of assigning layers 1-4 to GPU 0 and layers 5-8 to GPU 1, we assign:

- GPU 0: layers 1-2, 9-10
- GPU 1: layers 3-4, 11-12
- GPU 2: layers 5-6, 13-14
- GPU 3: layers 7-8, 15-16

This reduces the bubble from $\frac{(P-1)}{M}$ to $\frac{(P-1)}{M \cdot V}$ where $V$ is the number of stages per GPU (the "virtual pipeline stage" count). The cost is more frequent but smaller communication between stages.

### Communication Pattern

Pipeline parallelism uses **point-to-point** communication (send/recv) between adjacent stages. The volume per micro-batch is the activation tensor size at the boundary between stages:

$$\text{Volume} = b \times s \times h$$

where $b$ is the micro-batch size, $s$ is the sequence length, and $h$ is the hidden dimension. For a 7B model with $b=1$, $s=2048$, $h=4096$:

$$\text{Volume} = 1 \times 2048 \times 4096 \times 2 \text{ bytes (BF16)} = 16 \text{ MB per micro-batch}$$

This is relatively small compared to the all-reduce volume in data parallelism, making pipeline parallelism well-suited for **inter-node** communication.

### Pipeline Parallelism: Practical Considerations

**Memory imbalance:** The first and last stages of the pipeline often have different memory requirements than the middle stages. The first stage includes the embedding layer, and the last stage includes the output projection and loss computation. Careful layer assignment can balance memory across stages.

**Warmup and cooldown phases:** At the beginning of a mini-batch, only the first stage has work (the pipeline is "filling"). At the end, only the last stage has work (the pipeline is "draining"). These phases constitute the pipeline bubble. The 1F1B schedule minimizes the impact by keeping the steady state as long as possible.

**Weight update synchronization:** All stages must complete their gradient computations before the optimizer step. In synchronous pipeline parallelism, this means a global synchronization point at the end of each mini-batch.

**Activation memory management:** Each stage must store activations for all in-flight micro-batches (those that have passed through the stage's forward pass but have not yet completed their backward pass). This is typically the dominant memory cost in pipeline parallelism.

---

## 12.9 Sequence Parallelism

As context lengths grow from 2K to 128K tokens and beyond, **activation memory** becomes the dominant bottleneck. For a transformer layer, the activation memory for the attention computation scales as $O(b \times s^2 \times h)$ without Flash Attention, or $O(b \times s \times h)$ with Flash Attention — but even the linear scaling becomes prohibitive for very long sequences.

### The Idea

**Sequence parallelism** partitions the activation tensors along the sequence dimension across GPUs. Each GPU processes a contiguous chunk of the sequence.

In the Megatron-LM framework (Korthikanti et al., 2022), sequence parallelism is applied to the **non-tensor-parallel** regions of the transformer — specifically, the LayerNorm and dropout operations that follow the all-reduce in tensor parallelism:

```
Tensor Parallel region:          Sequence Parallel region:
[Column-parallel Linear] ──→     [All-Reduce] ──→ [LayerNorm + Dropout]
                                                         ↓
                                              (activations split along seq dim)
```

More precisely, the all-reduce in tensor parallelism is decomposed into a reduce-scatter (entering the sequence-parallel region) followed by an all-gather (leaving it). This keeps activation memory partitioned during the sequence-parallel regions without additional communication cost.

### Ring Attention

For very long sequences, **Ring Attention** (Liu et al., 2023) provides a way to parallelize the attention computation itself across the sequence dimension:

1. Partition the query, key, and value matrices along the sequence dimension across $N$ GPUs.
2. Each GPU computes attention between its local queries and all keys/values.
3. Keys and values are passed in a ring pattern — each GPU sends its KV shard to the next GPU and receives from the previous, computing partial attention scores at each step.
4. After $N$ steps, each GPU has computed attention over the full sequence.

The memory per GPU scales as $O(s/N)$ for the attention activations, enabling sequence lengths proportional to the number of GPUs.

### Why Sequence Parallelism Matters

For a 70B model with sequence length 128K and batch size 1:
- Without sequence parallelism: activation memory for a single layer's attention is approximately $128K \times 8192 \times 2 \times 2 = 4$ GB per layer, times 80 layers = 320 GB.
- With sequence parallelism across 8 GPUs: approximately 40 GB of activation memory, manageable with gradient checkpointing.

### Ring Attention Communication Pattern

The Ring Attention algorithm is particularly elegant in its communication design:

```
Step 1: GPU 0 computes attention(Q0, K0, V0)
        GPU 1 computes attention(Q1, K1, V1)
        GPU 2 computes attention(Q2, K2, V2)
        GPU 3 computes attention(Q3, K3, V3)
        [Simultaneously: each GPU sends its KV to the next GPU in the ring]

Step 2: GPU 0 computes attention(Q0, K3, V3) — received from GPU 3
        GPU 1 computes attention(Q1, K0, V0) — received from GPU 0
        GPU 2 computes attention(Q2, K1, V1) — received from GPU 1
        GPU 3 computes attention(Q3, K2, V2) — received from GPU 2
        [Simultaneously: pass KV to next GPU again]

... (N steps total, where N = number of GPUs)

Final: Each GPU has accumulated attention over the full sequence
```

The communication and computation are fully overlapped: while each GPU computes attention with the current KV block, it simultaneously sends that block to the next GPU and receives the next block from the previous GPU. The total communication volume equals the total KV size, which would need to be stored in memory anyway in a non-distributed setting.

### Sequence Parallelism vs. Context Parallelism

Some frameworks distinguish between **sequence parallelism** (as described in the Megatron-LM framework, applying to LayerNorm and dropout regions) and **context parallelism** (splitting the attention computation along the sequence dimension, as in Ring Attention). The former is a memory optimization within tensor-parallel groups, while the latter is a separate parallelism dimension that can be combined with other strategies.

---

## 12.10 Expert Parallelism for MoE Models

**Mixture of Experts (MoE)** models (Shazeer et al., 2017; Fedus et al., 2022) use a routing mechanism to activate only a subset of "expert" sub-networks for each input token. This increases model capacity without proportionally increasing compute cost.

### MoE Architecture Recap

In a standard transformer with MoE, the MLP in each layer is replaced by $E$ expert MLPs and a router:

$$y = \sum_{i=1}^{k} g_i \cdot \text{Expert}_{\text{top}_i}(x)$$

where $k$ is the number of experts activated per token (typically 1 or 2), and $g_i$ are the gating weights from the router.

### Expert Parallelism

With $E$ experts per layer, **expert parallelism** places different experts on different GPUs. If $E = 64$ and we have 8 GPUs, each GPU hosts 8 experts.

**Communication pattern — All-to-All:**

1. **Dispatch (all-to-all):** After routing, each GPU sends tokens to the GPU that hosts their assigned expert. This requires an all-to-all collective where each GPU sends different data to every other GPU.
2. **Expert computation:** Each GPU runs its local experts on the received tokens.
3. **Combine (all-to-all):** Results are sent back to the originating GPUs.

```
GPU 0: [tokens a,b,c,d] ──┐         ┌──→ [Expert 0,1 compute] ──┐         ┌──→ [results for a,b,c,d]
GPU 1: [tokens e,f,g,h] ──┤ All-to-All ├──→ [Expert 2,3 compute] ──┤ All-to-All ├──→ [results for e,f,g,h]
GPU 2: [tokens i,j,k,l] ──┤         ├──→ [Expert 4,5 compute] ──┤         ├──→ [results for i,j,k,l]
GPU 3: [tokens m,n,o,p] ──┘         └──→ [Expert 6,7 compute] ──┘         └──→ [results for m,n,o,p]
```

### Load Balancing

Expert parallelism introduces a unique challenge: if the router sends most tokens to a few experts, some GPUs will be overloaded while others are idle. Load balancing techniques include:

- **Auxiliary loss:** Add a loss term that penalizes uneven expert utilization (Fedus et al., 2022).
- **Expert capacity:** Cap the number of tokens each expert can process; overflow tokens are dropped or routed to a shared expert.
- **Token dropping:** Randomly drop tokens that exceed expert capacity during training.

### Communication Cost

The all-to-all volume per MoE layer is:

$$V = 2 \times b \times s \times h \times \text{bytes per element}$$

(factor of 2 for dispatch and combine). This is proportional to the total token count and hidden dimension, independent of the number of experts. However, all-to-all communication patterns are less bandwidth-efficient than all-reduce because the data is scattered across all GPUs rather than aggregated.

---

## 12.11 3D Parallelism — Putting It All Together

Real-world large-scale training combines multiple parallelism strategies. The canonical example is **Megatron-DeepSpeed** (Smith et al., 2022), which combines:

- **Tensor Parallelism (TP)** within a node
- **Pipeline Parallelism (PP)** across nodes
- **Data Parallelism (DP)** across pipeline-parallel groups

### Parallelism Configuration

For a cluster with $G$ total GPUs, organized into nodes of $g$ GPUs each:

$$G = \text{TP} \times \text{PP} \times \text{DP}$$

**Design principles:**

1. **Tensor parallelism first (intra-node):** Set TP to the number of GPUs per node (typically 4 or 8) connected via NVLink. This minimizes communication latency for the most frequent collective operations.

2. **Pipeline parallelism second (inter-node):** Set PP based on model depth and available inter-node bandwidth. PP communication is point-to-point and relatively small, making it suitable for lower-bandwidth interconnects.

3. **Data parallelism for the rest:** $\text{DP} = G / (\text{TP} \times \text{PP})$. Combine with ZeRO Stage 1 for additional memory savings.

### Example Configuration

**Scenario:** Train a 175B parameter model on 512 A100 GPUs (64 nodes, 8 GPUs per node).

Model dimensions: 96 layers, hidden size 12288, 96 attention heads.

**Configuration:**
- TP = 8 (one full node per tensor-parallel group)
- PP = 8 (8 pipeline stages, each with 12 layers, spanning 8 nodes)
- DP = 512 / (8 × 8) = 8 (8-way data parallelism)
- ZeRO Stage 1 across the 8 DP ranks

**Memory analysis per GPU:**
- Each GPU holds $175B / (8 \times 8) \approx 2.7B$ parameters (TP shards PP)
- With mixed precision: $2.7B \times 2 = 5.4$ GB for parameters
- Optimizer states with ZeRO-1: $2.7B \times 12 / 8 = 4.1$ GB
- Activations: managed via gradient checkpointing and sequence parallelism
- Total: well within 80 GB

**Communication pattern per training step:**

| Operation | Type | Frequency | Volume per GPU |
|---|---|---|---|
| TP all-reduce | Intra-node | 2 per layer per micro-batch | ~$2 \times h^2 \times 2$ bytes |
| PP send/recv | Inter-node | 1 per micro-batch | $b \times s \times h \times 2$ bytes |
| DP all-reduce | Inter-node | 1 per step | $2\Psi_\text{local}$ bytes |

### Practical Guidance for Choosing Parallelism

```python
def choose_parallelism(
    model_params_billions: float,
    gpus_per_node: int,
    num_nodes: int,
    gpu_memory_gb: float = 80,
    sequence_length: int = 2048,
):
    """Heuristic for choosing parallelism dimensions."""
    total_gpus = gpus_per_node * num_nodes

    # Memory per GPU needed (rough estimate, bytes)
    bytes_per_param = 18  # 2 (param) + 2 (grad) + 12 (optimizer) + 2 (buffer)
    model_memory_gb = model_params_billions * 1e9 * bytes_per_param / 1e9

    # Step 1: Determine minimum tensor parallelism
    # Each TP group must fit the model's largest layer in memory
    tp = 1
    while model_memory_gb / tp > gpu_memory_gb * 0.7:  # 70% of GPU memory
        tp *= 2
        if tp > gpus_per_node:
            break
    tp = min(tp, gpus_per_node)

    # Step 2: Determine pipeline parallelism
    memory_per_gpu_with_tp = model_memory_gb / tp
    pp = 1
    while memory_per_gpu_with_tp / pp > gpu_memory_gb * 0.5:
        pp *= 2

    # Step 3: Data parallelism is what remains
    dp = total_gpus // (tp * pp)

    print(f"Recommended: TP={tp}, PP={pp}, DP={dp}")
    print(f"  Total GPUs used: {tp * pp * dp}")
    print(f"  Memory per GPU: ~{model_memory_gb / (tp * pp):.1f} GB")

    return tp, pp, dp

# Example: 70B model on 32 A100-80GB GPUs (4 nodes of 8)
choose_parallelism(70, 8, 4)
# Output: Recommended: TP=8, PP=4, DP=1
```

### The Communication Hierarchy

The guiding principle behind 3D parallelism is to **match the parallelism strategy to the communication bandwidth**:

```
Highest bandwidth ──→ NVLink (600-900 GB/s) ──→ Tensor Parallelism
Medium bandwidth  ──→ InfiniBand (25-100 GB/s) ──→ Pipeline / Data Parallelism
Lowest bandwidth  ──→ Ethernet (1-10 GB/s)     ──→ Data Parallelism (with ZeRO)
```

This hierarchy ensures that the most communication-intensive strategies (tensor parallelism with per-layer all-reduces) use the fastest interconnect, while less frequent but larger communications (gradient all-reduce) can tolerate lower bandwidth.

---

## Exercises

1. **Memory Calculation:** A 13B parameter model is trained with AdamW in BF16 mixed precision. Calculate the per-GPU memory requirement (excluding activations) for: (a) standard DDP on 8 GPUs, (b) FSDP FULL_SHARD on 8 GPUs, (c) DeepSpeed ZeRO Stage 2 on 8 GPUs.

2. **Communication Analysis:** For a ring all-reduce of a 1 GB gradient tensor across 16 GPUs, calculate: (a) the per-GPU data transfer volume, (b) the time to complete on NVLink at 600 GB/s, (c) the time to complete on InfiniBand at 50 GB/s.

3. **Gradient Accumulation:** You are training with a micro-batch size of 2 per GPU, 8 GPUs, and gradient accumulation of 16 steps. What is the effective batch size? If the sequence length is 4096 tokens, how many tokens per gradient update?

4. **Pipeline Bubble:** A 4-stage pipeline processes 32 micro-batches. Calculate the bubble fraction. What is the bubble fraction with interleaved scheduling using 2 virtual stages per GPU?

5. **3D Parallelism Design:** You have 256 H100 GPUs (32 nodes of 8 GPUs). Design a 3D parallelism configuration for training a 405B parameter model with 126 layers, hidden size 16384, and 128 attention heads. Justify your choices.

6. **Implementation:** Implement a DDP training script using `torchrun` that trains a simple transformer model on a synthetic dataset. Include gradient accumulation, distributed sampling, and logging that prints loss only on rank 0.

7. **FSDP vs. ZeRO:** Compare PyTorch FSDP and DeepSpeed ZeRO Stage 3 in terms of: (a) communication primitives used, (b) memory savings, (c) integration complexity with Hugging Face models, (d) checkpoint saving/loading patterns.

---

## References

Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23(120), 1-39.

Huang, Y., Cheng, Y., Bapna, A., Firat, O., Chen, D., Chen, M., ... & Wu, Y. (2019). GPipe: Efficient training of giant neural networks using pipeline parallelism. *Advances in Neural Information Processing Systems*, 32.

Jeaugey, S. (2017). NCCL 2.0. *GPU Technology Conference (GTC)*.

Korthikanti, V. A., Casper, J., Lym, S., McAfee, L., Andersch, M., Shoeybi, M., & Catanzaro, B. (2022). Reducing activation recomputation in large transformer models. *arXiv preprint arXiv:2205.05198*.

Liu, H., Zaharia, M., & Abbeel, P. (2023). Ring Attention with blockwise transformers for near-infinite context. *arXiv preprint arXiv:2310.01889*.

Narayanan, D., Harlap, A., Phanishayee, A., Seshadri, V., Devanur, N. R., Ganger, G. R., ... & Zaharia, M. (2019). PipeDream: Generalized pipeline parallelism for DNN training. *Proceedings of the 27th ACM Symposium on Operating Systems Principles*, 1-15.

Narayanan, D., Shoeybi, M., Casper, J., LeGresley, P., Patwary, M., Korthikanti, V., ... & Catanzaro, B. (2021). Efficient large-scale language model training on GPU clusters using Megatron-LM. *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis*, 1-15.

Rajbhandari, S., Rasley, J., Rber, O., & He, Y. (2020). ZeRO: Memory optimizations toward training trillion parameter models. *International Conference for High Performance Computing, Networking, Storage and Analysis (SC)*.

Rajbhandari, S., Rber, O., Rasley, J., & He, Y. (2021). ZeRO-Infinity: Breaking the GPU memory wall for extreme scale deep learning. *International Conference for High Performance Computing, Networking, Storage and Analysis (SC)*.

Sevilla, J., Heim, L., Ho, A., Besiroglu, T., Hobbhahn, M., & Villalobos, P. (2022). Compute trends across three eras of machine learning. *2022 International Joint Conference on Neural Networks (IJCNN)*, 1-8.

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). Outrageously large neural networks: The sparsely-gated Mixture-of-Experts layer. *International Conference on Learning Representations*.

Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., & Catanzaro, B. (2019). Megatron-LM: Training multi-billion parameter language models using model parallelism. *arXiv preprint arXiv:1909.08053*.

Smith, S., Patwary, M., Norick, B., LeGresley, P., Rajbhandari, S., Casper, J., ... & Catanzaro, B. (2022). Using DeepSpeed and Megatron to train Megatron-Turing NLG 530B, a large-scale generative language model. *arXiv preprint arXiv:2201.11990*.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., ... & Scialom, T. (2023). Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*.

Zhao, Y., Gu, A., Varma, R., Luo, L., Huang, C. C., Xu, M., ... & Gimelshein, N. (2023). PyTorch FSDP: Experiences on scaling fully sharded data parallel. *Proceedings of the VLDB Endowment*, 16(12), 3848-3860.
