# Chapter 18: LLM Serving and Inference

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Articulate the serving challenge** — define TTFT, TBT, throughput, and cost metrics, and explain why LLM serving is fundamentally different from traditional ML serving.
2. **Package and deploy models with BentoML** — write bentofile configurations, define Service classes with runners, configure adaptive batching, and containerize for deployment.
3. **Deploy high-throughput inference with vLLM** — explain PagedAttention, implement continuous batching, configure tensor parallelism, and benchmark an OpenAI-compatible API server.
4. **Serve models with TGI** — configure Hugging Face's Text Generation Inference with Flash Attention, quantization, streaming, and watermarking.
5. **Optimize models with ONNX and TensorRT** — export PyTorch models to ONNX, optimize with TensorRT, handle dynamic axes, and perform INT8 calibration.
6. **Manage model serving with Triton Inference Server** — configure model repositories, dynamic batching, model ensembles, and concurrent execution.
7. **Deploy quantized models for edge and production** — apply GPTQ, AWQ, and GGUF quantization for different deployment targets.
8. **Implement advanced serving techniques** — speculative decoding, LoRA serving with S-LoRA, structured generation with SGLang, KV cache optimization, and cost optimization strategies.

---

## 18.1 The Serving Challenge

### 18.1.1 Key Metrics

LLM serving introduces metrics that traditional ML serving does not encounter:

**Time to First Token (TTFT).** The latency from receiving a request to generating the first output token. This corresponds to the **prefill phase**, where the model processes all input tokens and builds the KV cache. TTFT is critical for interactive applications — users perceive delays > 500ms as sluggish.

**Time Between Tokens (TBT).** The latency between consecutive output tokens during the **decode phase**. Each decode step generates one token by attending to all previous KV cache entries. For streaming applications, TBT determines the perceived fluency of generation. Human reading speed is approximately 250 words per minute, or ~6 tokens per second, so TBT must stay below ~160ms for the output to feel smooth.

**Throughput.** The total number of tokens generated per second across all concurrent requests. Throughput determines how many users a single GPU can serve simultaneously.

**Cost per token.** The total infrastructure cost divided by the number of tokens served. This is the ultimate business metric — it determines the unit economics of an LLM-powered product.

**End-to-end latency.** The total time from request to complete response. For non-streaming use cases, this is what matters: $\text{E2E} = \text{TTFT} + \text{TBT} \times n_{\text{output tokens}}$.

### 18.1.2 Why LLM Serving Is Different

Traditional ML serving (classification, regression, object detection) processes fixed-size inputs and produces fixed-size outputs in a single forward pass. LLM serving is fundamentally different in several ways:

**Autoregressive generation.** Output tokens are generated one at a time, with each token depending on all previous tokens. A 500-token response requires 500 sequential forward passes — the generation process is inherently serial.

**Variable sequence lengths.** Both inputs and outputs vary in length, from a few tokens to hundreds of thousands. This makes batching non-trivial: naive batching pads all sequences to the maximum length, wasting computation.

**Memory-bound decode.** During the decode phase, each forward pass reads the entire KV cache but produces only one token. The arithmetic intensity (FLOPs per byte of memory access) is extremely low, making the decode phase memory-bandwidth-bound rather than compute-bound. This is the opposite of training, which is compute-bound.

**KV cache memory.** The KV cache grows linearly with sequence length and batch size. For a 70B parameter model with 128K context, the KV cache alone can require hundreds of gigabytes of GPU memory. Managing this memory efficiently is the central challenge of LLM serving.

**Cost structure.** GPU compute costs dominate. A single A100 GPU costs ~$2/hour; serving a 70B model requires 2–4 A100s. If those GPUs serve 100 requests per minute, the cost per request is $0.001–0.003 — competitive only if GPU utilization is high.

---

## 18.2 BentoML

### 18.2.1 Overview

BentoML (Yang et al., 2024) is a model serving framework that abstracts the complexity of packaging, containerizing, and deploying ML models. It provides a Python-first API for defining inference services with built-in support for batching, concurrency, and monitoring.

### 18.2.2 Model Packaging

A BentoML service is defined by a `bentofile.yaml` configuration and a Python service file:

```yaml
# bentofile.yaml
service: "service:LLMService"
labels:
  owner: ml-team
  project: llm-serving
include:
  - "*.py"
python:
  requirements_txt: "./requirements.txt"
docker:
  python_version: "3.11"
  system_packages:
    - "build-essential"
```

### 18.2.3 Service Class

```python
# service.py
import bentoml
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

@bentoml.service(
    resources={"gpu": 1, "memory": "32Gi"},
    traffic={"timeout": 120, "concurrency": 32}
)
class LLMService:
    def __init__(self):
        self.model_id = "meta-llama/Llama-3.1-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=8,
        max_latency_ms=1000
    )
    async def generate(self, prompts: list[str]) -> list[str]:
        """Generate text for a batch of prompts."""
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )

        responses = self.tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return responses
```

### 18.2.4 Adaptive Batching

BentoML's adaptive batching dynamically adjusts batch sizes based on incoming request patterns. The `@bentoml.api(batchable=True)` decorator enables this:

- **`max_batch_size`:** Upper limit on batch size. Larger batches improve throughput but increase latency for individual requests.
- **`max_latency_ms`:** Maximum time to wait for the batch to fill before processing. Lower values reduce latency; higher values improve batching efficiency.

The system automatically balances these parameters based on real-time traffic, collecting requests into batches that maximize throughput while respecting latency constraints.

### 18.2.5 Containerization and Deployment

```bash
# Build the Bento (packaged artifact)
bentoml build

# Containerize
bentoml containerize llm-service:latest

# Deploy (example: BentoCloud)
bentoml deploy llm-service:latest
```

BentoML containers include all dependencies, model weights, and serving infrastructure. They can be deployed to any container orchestration platform (Kubernetes, ECS, Cloud Run) or to BentoCloud for managed serving.

---

## 18.3 vLLM

### 18.3.1 PagedAttention

vLLM (Kwon et al., 2023) introduced **PagedAttention**, the most significant innovation in LLM serving. The key insight: KV cache memory management in traditional serving systems is wasteful because sequences are allocated contiguous memory blocks sized for the maximum possible sequence length, even when the actual sequence is much shorter.

PagedAttention borrows the concept of **virtual memory and paging** from operating systems. Instead of allocating contiguous memory for each sequence's KV cache:

1. KV cache memory is divided into fixed-size **blocks** (pages), each holding KV values for a fixed number of tokens (e.g., 16 tokens per block).
2. Each sequence maintains a **block table** mapping logical KV cache positions to physical memory blocks.
3. Blocks are allocated on demand as the sequence grows, and freed when the sequence completes.
4. Multiple sequences can **share blocks** (for shared prefixes, beam search candidates, or parallel sampling), with copy-on-write semantics.

This approach reduces KV cache memory waste from ~60–80% (in naive implementations) to near zero, enabling 2–4x higher throughput by fitting more concurrent sequences in the same GPU memory.

### 18.3.2 Continuous Batching

Traditional batching waits for an entire batch to complete before processing new requests. Since sequences have varying lengths, short sequences are delayed by long ones. **Continuous batching** (also called iteration-level batching) addresses this:

- At each decode step, completed sequences are evicted from the batch and new pending requests are inserted.
- This keeps the GPU busy with a full batch at every iteration, maximizing utilization.
- Combined with PagedAttention, continuous batching enables near-optimal GPU utilization.

### 18.3.3 Tensor Parallelism for Serving

For models too large to fit on a single GPU, vLLM supports tensor parallelism — splitting model layers across GPUs:

```bash
# Serve a 70B model across 4 GPUs with tensor parallelism
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --dtype float16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --port 8000
```

### 18.3.4 OpenAI-Compatible API Server

vLLM provides an OpenAI-compatible API server, enabling drop-in replacement of OpenAI's API with a self-hosted model:

```python
from openai import OpenAI

# Point to vLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # vLLM doesn't require an API key
)

# Use exactly like OpenAI API
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-70B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain PagedAttention in simple terms."}
    ],
    temperature=0.7,
    max_tokens=512,
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 18.3.5 Benchmarking

```python
import asyncio
import aiohttp
import time
import numpy as np

async def benchmark_vllm(
    url: str = "http://localhost:8000/v1/completions",
    num_requests: int = 100,
    concurrency: int = 10,
    prompt: str = "Explain the theory of relativity in detail.",
    max_tokens: int = 256
):
    """Benchmark vLLM server throughput and latency."""
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def send_request(session):
        async with semaphore:
            start = time.time()
            async with session.post(url, json={
                "model": "meta-llama/Llama-3.1-70B-Instruct",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7
            }) as resp:
                data = await resp.json()
                elapsed = time.time() - start
                tokens = data["usage"]["completion_tokens"]
                results.append({"latency": elapsed, "tokens": tokens})

    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        tasks = [send_request(session) for _ in range(num_requests)]
        await asyncio.gather(*tasks)
        total_time = time.time() - start_time

    latencies = [r["latency"] for r in results]
    total_tokens = sum(r["tokens"] for r in results)

    print(f"Requests: {num_requests}")
    print(f"Concurrency: {concurrency}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {total_tokens / total_time:.1f} tokens/sec")
    print(f"Latency P50: {np.percentile(latencies, 50):.3f}s")
    print(f"Latency P99: {np.percentile(latencies, 99):.3f}s")

asyncio.run(benchmark_vllm())
```

---

## 18.4 TGI (Text Generation Inference)

### 18.4.1 Overview

Text Generation Inference (TGI) is Hugging Face's production-grade LLM serving solution. Written in Rust and Python, TGI provides:

- **Flash Attention** integration for efficient attention computation.
- **Continuous batching** (similar to vLLM).
- **Quantization** support (GPTQ, AWQ, bitsandbytes).
- **Streaming** via Server-Sent Events (SSE).
- **Watermarking** to detect AI-generated text.
- **Token-level streaming** with logprobs.

### 18.4.2 Deployment

```bash
# Deploy with Docker
docker run --gpus all --shm-size 1g -p 8080:80 \
    -v $PWD/data:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --quantize awq \
    --max-input-length 4096 \
    --max-total-tokens 8192 \
    --max-batch-prefill-tokens 4096 \
    --max-concurrent-requests 128
```

### 18.4.3 Flash Attention Integration

TGI integrates Flash Attention (Dao et al., 2022; Dao, 2023) to accelerate the attention computation:

- **IO-aware:** Flash Attention minimizes memory reads/writes between GPU SRAM and HBM by fusing the softmax computation with the matrix multiplication, processing attention in tiles.
- **Memory efficient:** Standard attention materializes the $N \times N$ attention matrix ($O(N^2)$ memory); Flash Attention avoids this, using $O(N)$ memory.
- **Speed:** 2–4x faster than standard attention, with the speedup increasing for longer sequences.

### 18.4.4 Streaming and Watermarking

TGI supports token-level streaming via SSE:

```python
import requests
import json

def stream_tgi(prompt: str, url: str = "http://localhost:8080/generate_stream"):
    """Stream tokens from TGI."""
    response = requests.post(
        url,
        json={
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "watermark": True,  # Enable AI text watermarking
                "return_full_text": False
            }
        },
        stream=True
    )

    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8").removeprefix("data:"))
            token = data.get("token", {}).get("text", "")
            print(token, end="", flush=True)

            if data.get("generated_text"):
                print()  # Final newline
                break
```

TGI's watermarking implementation follows the approach of Kirchenbauer et al. (2023), subtly biasing token selection to embed a detectable signal that does not affect text quality.

---

## 18.5 ONNX + TensorRT

### 18.5.1 Export Pipeline

The ONNX-TensorRT pipeline converts PyTorch models into optimized inference engines:

**Step 1: PyTorch to ONNX.** Export the model to ONNX (Open Neural Network Exchange), an open format for representing ML models.

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create dummy input
dummy_input = tokenizer(
    "This is a test sentence",
    return_tensors="pt",
    padding="max_length",
    max_length=128,
    truncation=True
)

# Export to ONNX
torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    "model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"}
    },
    opset_version=17
)
```

**Step 2: ONNX optimization.** Use ONNX Runtime's optimization tools:

```python
from onnxruntime.transformers import optimizer

optimized_model = optimizer.optimize_model(
    "model.onnx",
    model_type="bert",
    num_heads=12,
    hidden_size=768,
    optimization_options=None
)
optimized_model.save_model_to_file("model_optimized.onnx")
```

**Step 3: TensorRT optimization.** Convert the ONNX model to a TensorRT engine for maximum GPU performance:

```python
import tensorrt as trt

def build_engine(onnx_path: str, engine_path: str,
                 precision: str = "fp16") -> None:
    """Build a TensorRT engine from an ONNX model."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("ONNX parsing failed")

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        # INT8 requires calibration data
        config.int8_calibrator = EntropyCalibrator(calibration_data)

    # Set dynamic shapes
    profile = builder.create_optimization_profile()
    profile.set_shape("input_ids", (1, 1), (8, 64), (32, 128))
    profile.set_shape("attention_mask", (1, 1), (8, 64), (32, 128))
    config.add_optimization_profile(profile)

    # Build engine
    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
```

### 18.5.2 Dynamic Axes and Operator Support

**Dynamic axes** allow the TensorRT engine to handle variable-length inputs. The profile specifies minimum, optimal, and maximum shapes for each dynamic dimension. TensorRT builds optimized kernels for shapes near the optimal, with fallback for other shapes.

**Operator support** can be a challenge. Not all PyTorch operations have ONNX equivalents, and not all ONNX operations are supported by TensorRT. Common issues include:
- Custom attention patterns (Flash Attention, grouped-query attention)
- Dynamic control flow (if/else based on tensor values)
- Certain activation functions or normalization layers

Workarounds include custom ONNX operator registration, TensorRT plugin development, and model modification to avoid unsupported operations.

### 18.5.3 INT8 Calibration

INT8 quantization in TensorRT requires calibration — running representative data through the model to determine the optimal quantization ranges for each layer:

```python
import tensorrt as trt
import numpy as np

class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_loader, cache_file="calibration.cache"):
        super().__init__()
        self.data_loader = iter(data_loader)
        self.cache_file = cache_file
        self.batch_size = 8

        # Allocate device memory for calibration data
        self.device_input = cuda.mem_alloc(
            self.batch_size * 128 * np.dtype(np.int32).itemsize
        )

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        try:
            batch = next(self.data_loader)
            cuda.memcpy_htod(self.device_input, batch.numpy())
            return [int(self.device_input)]
        except StopIteration:
            return None

    def read_calibration_cache(self):
        try:
            with open(self.cache_file, "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
```

---

## 18.6 Triton Inference Server

### 18.6.1 Overview

NVIDIA Triton Inference Server provides a standardized platform for serving models from multiple frameworks (TensorRT, ONNX Runtime, PyTorch, TensorFlow, vLLM) with enterprise-grade features.

### 18.6.2 Model Repository

Triton organizes models in a repository with a specific directory structure:

```
model_repository/
├── llm/
│   ├── config.pbtxt
│   └── 1/
│       └── model.plan          # TensorRT engine
├── tokenizer/
│   ├── config.pbtxt
│   └── 1/
│       └── model.py            # Python backend
└── ensemble/
    └── config.pbtxt            # Ensemble pipeline
```

```protobuf
# config.pbtxt for the LLM model
name: "llm"
backend: "tensorrt_llm"
max_batch_size: 32

input [
  {
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [-1]  # Dynamic sequence length
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT32
    dims: [-1]
  }
]

output [
  {
    name: "logits"
    data_type: TYPE_FP32
    dims: [-1, -1]
  }
]

dynamic_batching {
  preferred_batch_size: [4, 8, 16, 32]
  max_queue_delay_microseconds: 100000
}

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [0]
  }
]
```

### 18.6.3 Dynamic Batching

Triton's dynamic batching combines individual inference requests into batches on the server side:

- **`preferred_batch_size`:** Target batch sizes that the scheduler will try to form.
- **`max_queue_delay_microseconds`:** Maximum time to wait for more requests to fill a preferred batch size.
- **Priority levels:** Higher-priority requests can preempt lower-priority ones.

### 18.6.4 Model Ensembles

Triton supports model ensembles — pipelines where the output of one model feeds into the input of another:

```protobuf
# config.pbtxt for ensemble
name: "ensemble"
platform: "ensemble"
max_batch_size: 32

input [
  {
    name: "text"
    data_type: TYPE_STRING
    dims: [1]
  }
]

output [
  {
    name: "generated_text"
    data_type: TYPE_STRING
    dims: [1]
  }
]

ensemble_scheduling {
  step [
    {
      model_name: "tokenizer"
      model_version: 1
      input_map {
        key: "text"
        value: "text"
      }
      output_map {
        key: "input_ids"
        value: "input_ids"
      }
    },
    {
      model_name: "llm"
      model_version: 1
      input_map {
        key: "input_ids"
        value: "input_ids"
      }
      output_map {
        key: "logits"
        value: "logits"
      }
    }
  ]
}
```

### 18.6.5 Concurrent Model Execution and Metrics

Triton can serve multiple models concurrently on the same GPU, time-slicing between them. It exposes Prometheus-compatible metrics for monitoring:

- Request count and latency (per model)
- Queue depth and wait time
- GPU utilization and memory usage
- Inference throughput (inferences/second)

---

## 18.7 Quantization for Deployment

### 18.7.1 GPTQ

GPTQ (Frantar et al., 2023) performs post-training quantization by solving a layer-wise reconstruction problem. For each layer, GPTQ finds INT4 weights $\hat{\mathbf{W}}$ that minimize:

$$\|\mathbf{W}\mathbf{X} - \hat{\mathbf{W}}\mathbf{X}\|_2^2$$

where $\mathbf{X}$ is a calibration dataset. GPTQ processes columns of the weight matrix sequentially, using Hessian information to determine the optimal quantization order and compensate for quantization error in subsequent columns.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Configure GPTQ quantization
quantization_config = GPTQConfig(
    bits=4,
    dataset="c4",          # Calibration dataset
    tokenizer=tokenizer,
    group_size=128,         # Quantize in groups of 128
    desc_act=True           # Descending activation order
)

# Quantize model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"
)

# Save quantized model
model.save_pretrained("llama-8b-gptq-4bit")
```

### 18.7.2 AWQ

AWQ (Activation-aware Weight Quantization; Lin et al., 2024) observes that not all weight channels are equally important — channels corresponding to large activation magnitudes contribute disproportionately to model quality. AWQ protects these "salient" channels by scaling them up before quantization:

$$\hat{\mathbf{w}} = \text{Quant}(\mathbf{w} \cdot s), \quad \hat{\mathbf{x}} = \mathbf{x} / s$$

where $s$ is a per-channel scaling factor derived from activation statistics. This simple scaling trick significantly improves quantization quality compared to naive round-to-nearest quantization.

AWQ typically outperforms GPTQ at the same bit-width, particularly for smaller models where each weight's contribution is more significant.

### 18.7.3 GGUF and llama.cpp for Edge Deployment

**GGUF** (GPT-Generated Unified Format) is the file format used by **llama.cpp**, a C/C++ implementation of LLM inference that runs on CPUs, Apple Silicon (Metal), and consumer GPUs without requiring CUDA.

llama.cpp supports multiple quantization levels:
- **Q8_0:** 8-bit quantization (~50% size reduction, minimal quality loss)
- **Q5_K_M:** 5-bit with k-quants (~65% size reduction, small quality loss)
- **Q4_K_M:** 4-bit with k-quants (~75% size reduction, moderate quality loss)
- **Q3_K_M:** 3-bit (~80% size reduction, noticeable quality loss)
- **Q2_K:** 2-bit (~85% size reduction, significant quality loss)

```bash
# Convert a model to GGUF format and quantize
python convert_hf_to_gguf.py meta-llama/Llama-3.1-8B-Instruct \
    --outfile llama-8b-f16.gguf

# Quantize to 4-bit
./llama-quantize llama-8b-f16.gguf llama-8b-q4_k_m.gguf Q4_K_M

# Run inference
./llama-cli -m llama-8b-q4_k_m.gguf \
    -p "Explain quantum computing" \
    -n 256 \
    --ctx-size 4096 \
    --threads 8
```

GGUF enables running LLMs on laptops, phones, and edge devices, democratizing access to local LLM inference.

### 18.7.4 ExLlamaV2 for GPTQ Inference

ExLlamaV2 is a highly optimized GPTQ inference engine that achieves significantly higher throughput than standard Hugging Face GPTQ inference through custom CUDA kernels:

```python
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2Sampler

# Load GPTQ model with ExLlamaV2
config = ExLlamaV2Config("llama-8b-gptq-4bit/")
model = ExLlamaV2(config)
model.load()
tokenizer = ExLlamaV2Tokenizer(config)

# Configure sampling
settings = ExLlamaV2Sampler.Settings()
settings.temperature = 0.7
settings.top_p = 0.9
settings.top_k = 40

# Stream generation
generator = ExLlamaV2StreamingGenerator(model, tokenizer)
generator.set_stop_conditions([tokenizer.eos_token_id])

input_ids = tokenizer.encode("Explain PagedAttention:")
generator.begin_stream_ex(input_ids, settings)

while True:
    result = generator.stream_ex()
    if result["eos"]:
        break
    print(result["chunk"], end="", flush=True)
```

---

## 18.8 Speculative Decoding in Production

### 18.8.1 The Mechanism

Speculative decoding (Leviathan et al., 2023; Chen et al., 2023) accelerates inference by using a small, fast **draft model** to generate candidate tokens, which are then verified in parallel by the large **target model**.

The algorithm:
1. The draft model generates $K$ candidate tokens autoregressively (fast, since it is small).
2. The target model processes all $K$ candidates in a single forward pass (parallel verification).
3. Accepted tokens (where the target model agrees) are kept; the first rejected token is resampled from the target model's distribution.

The key property is that speculative decoding produces **exactly the same distribution** as sampling directly from the target model — it is a mathematically lossless acceleration technique.

### 18.8.2 Choosing Draft Models

The draft model should be:
- **Fast:** Much smaller than the target model (typically 10–20% of parameters).
- **Aligned:** Trained on similar data and with similar tokenization as the target model.
- **Accepting:** High acceptance rate (> 70% of draft tokens accepted by the target).

Common pairings:
| Target Model | Draft Model | Typical Acceptance Rate |
|-------------|-------------|------------------------|
| Llama-3.1-70B | Llama-3.1-8B | 70–80% |
| GPT-4 | GPT-3.5-turbo | 60–75% |
| Mixtral-8x22B | Mixtral-8x7B | 65–80% |

### 18.8.3 When Speculative Decoding Helps

Speculative decoding provides the greatest speedup when:
- The decode phase dominates latency (long output sequences).
- The draft model has a high acceptance rate (> 70%).
- The target model is memory-bandwidth-bound (large model, small batch size).
- GPU compute is underutilized during normal decoding.

Speedups of 2–3x are common in practice. The technique is less effective for large batch sizes (where the target model's decode is already compute-saturated) or very short outputs.

```python
# Speculative decoding with vLLM
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_model="meta-llama/Llama-3.1-8B-Instruct",
    num_speculative_tokens=5,  # Draft 5 tokens at a time
    tensor_parallel_size=4
)

params = SamplingParams(temperature=0.7, max_tokens=512)
outputs = llm.generate(["Explain the theory of relativity."], params)
```

---

## 18.9 LoRA Serving: S-LoRA

### 18.9.1 The Multi-Tenant Challenge

Many organizations need to serve multiple fine-tuned variants of the same base model — one per customer, one per task, or one per department. Loading separate model copies for each variant wastes GPU memory and limits the number of concurrent variants.

### 18.9.2 S-LoRA Approach

S-LoRA (Sheng et al., 2023) solves this by maintaining a single base model in GPU memory and dynamically loading LoRA adapters per request:

**Architecture:**
- The base model weights $\mathbf{W}_0$ reside permanently in GPU memory.
- Each LoRA adapter $(\mathbf{A}_i, \mathbf{B}_i)$ is small (typically < 0.1% of base model parameters).
- At inference time, the forward pass computes: $\mathbf{h} = (\mathbf{W}_0 + \mathbf{B}_i \mathbf{A}_i) \mathbf{x}$

**Memory hierarchy:**
- **GPU memory:** Base model + frequently used adapters (hot cache).
- **CPU memory:** Less frequently used adapters (warm cache).
- **Disk:** Rarely used adapters (cold storage).

S-LoRA uses a custom CUDA kernel to **batch** LoRA computations across requests with different adapters, maintaining high throughput even when each request uses a different adapter.

### 18.9.3 Multi-Tenant Serving

```python
# Multi-tenant LoRA serving with vLLM
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_lora=True,
    max_loras=16,                    # Max concurrent adapters
    max_lora_rank=64,                # Max LoRA rank
    max_cpu_loras=64                 # Adapters cached in CPU memory
)

# Serve different customers with different adapters
params = SamplingParams(temperature=0.7, max_tokens=256)

# Customer A: legal domain adapter
output_a = llm.generate(
    ["Summarize this contract clause:"],
    params,
    lora_request=LoRARequest("legal_adapter", 1, "adapters/legal/")
)

# Customer B: medical domain adapter
output_b = llm.generate(
    ["Explain this diagnosis:"],
    params,
    lora_request=LoRARequest("medical_adapter", 2, "adapters/medical/")
)

# Customer C: no adapter (base model)
output_c = llm.generate(
    ["Tell me a joke:"],
    params
)
```

### 18.9.4 Adapter Management

Production LoRA serving requires adapter lifecycle management:
- **Versioning:** Track adapter versions alongside training runs.
- **A/B testing:** Route a percentage of traffic to new adapter versions.
- **Monitoring:** Track per-adapter metrics (latency, quality, error rates).
- **Garbage collection:** Evict unused adapters from GPU/CPU memory.
- **Hot-swapping:** Update adapters without restarting the server.

---

## 18.10 SGLang

### 18.10.1 Structured Generation

SGLang (Zheng et al., 2024) addresses a critical production need: generating structured output (JSON, XML, code) that conforms to a specified schema. Traditional sampling can produce malformed output; SGLang guarantees structural validity through constrained decoding.

### 18.10.2 Constrained Decoding

SGLang implements constrained decoding by modifying the token sampling distribution at each generation step. At each step, only tokens that are consistent with the target schema receive non-zero probability:

**JSON mode:** The model is constrained to output valid JSON matching a specified JSON Schema. At each token position, SGLang computes which tokens are valid given the current parse state and masks all others.

**Regex constraints:** The output must match a specified regular expression. SGLang compiles the regex into a finite automaton and tracks the current state during generation.

```python
import sglang as sgl

@sgl.function
def extract_info(s, text):
    s += sgl.system("You are an information extraction assistant.")
    s += sgl.user(f"Extract the following information from this text:\n{text}")
    s += sgl.assistant(
        sgl.gen("result",
                max_tokens=256,
                regex=r'\{"name": "[^"]+", "age": \d+, "city": "[^"]+"\}')
    )

# Run with constrained generation
state = extract_info.run(
    text="John Smith, a 32-year-old engineer from San Francisco...",
    backend=sgl.RuntimeEndpoint("http://localhost:30000")
)
print(state["result"])
# Guaranteed valid JSON: {"name": "John Smith", "age": 32, "city": "San Francisco"}
```

### 18.10.3 RadixAttention for Prefix Caching

SGLang introduces **RadixAttention**, an efficient prefix caching mechanism that stores KV caches in a radix tree data structure. When multiple requests share a common prefix (system prompt, few-shot examples, shared context), their KV cache entries are computed once and shared:

```
Radix Tree:
            [System Prompt KV Cache]
           /                        \
    [User A context]          [User B context]
    /           \                    |
[Query 1]  [Query 2]          [Query 3]
```

This eliminates redundant computation for shared prefixes. In production systems where many requests share the same system prompt (often 500–2000 tokens), RadixAttention can reduce TTFT by 50–80%.

---

## 18.11 KV Cache Optimization

### 18.11.1 Prefix Caching

When many requests share a common prefix (system prompt, instructions, shared context), the KV cache for that prefix can be computed once and reused:

```python
# vLLM prefix caching
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_prefix_caching=True  # Enable automatic prefix caching
)

# The system prompt KV cache is computed once and reused
system_prompt = "You are a helpful customer service agent for Acme Corp..."

requests = [
    f"{system_prompt}\nUser: How do I return an item?\nAssistant:",
    f"{system_prompt}\nUser: What is your refund policy?\nAssistant:",
    f"{system_prompt}\nUser: Can I change my shipping address?\nAssistant:",
]

params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(requests, params)
# First request computes system prompt KV cache
# Subsequent requests reuse it, reducing TTFT
```

### 18.11.2 Prompt Caching (API-Level)

Cloud providers (Anthropic, OpenAI) offer API-level prompt caching that automatically caches the KV state of repeated prompt prefixes:

- **Anthropic prompt caching:** Explicitly mark cacheable sections of the prompt. Cached prefixes are served at 90% lower cost and with lower latency.
- **OpenAI automatic caching:** Automatically caches the longest matching prefix across requests.

```python
import anthropic

client = anthropic.Anthropic()

# First request: computes and caches the system prompt
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "You are an expert on the company's 500-page policy manual...",
            "cache_control": {"type": "ephemeral"}  # Mark as cacheable
        }
    ],
    messages=[{"role": "user", "content": "What is the vacation policy?"}]
)

# Subsequent requests with same system prompt use cached KV state
# Cost: ~10% of original for the cached portion
```

### 18.11.3 Chunked Prefill

**Chunked prefill** splits long input sequences into smaller chunks and interleaves prefill computation with decode steps from other requests. This prevents long prompts from monopolizing the GPU and causing latency spikes for concurrent shorter requests:

1. A request with 10,000 input tokens arrives.
2. Instead of processing all 10,000 tokens in one prefill step (blocking all other requests for ~1 second), the system processes 2,000 tokens at a time.
3. Between prefill chunks, decode steps for other active requests are processed.
4. This smooths out latency across all concurrent requests.

vLLM and TGI both support chunked prefill, with configurable chunk sizes that balance prefill throughput against decode latency impact.

---

## 18.12 Cost Optimization

### 18.12.1 Model Selection

The most impactful cost optimization is choosing the right model size. A smaller model with better prompting often outperforms a larger model with naive prompting:

| Strategy | Model | Cost (per 1M tokens) | Quality (MMLU) |
|----------|-------|---------------------|-----------------|
| Large, simple | GPT-4o | $5.00 input / $15.00 output | 88% |
| Medium, optimized | Claude 3.5 Sonnet | $3.00 / $15.00 | 89% |
| Small, heavily prompted | Llama-3.1-8B + RAG | $0.10 (self-hosted) | 82% + RAG grounding |
| Tiny, specialized | Fine-tuned Llama-3.1-3B | $0.03 (self-hosted) | 85% (on-domain) |

For many production use cases, a fine-tuned small model outperforms a general-purpose large model at a fraction of the cost.

### 18.12.2 Batching Strategies

Efficient batching is crucial for self-hosted models:

**Static batching:** Collect $N$ requests, process together. Simple but suboptimal — all requests must wait for the longest one to complete.

**Continuous batching:** Insert and evict requests at each decode step. Maximizes GPU utilization. Implemented by vLLM and TGI.

**Padding-free batching:** Pack variable-length sequences contiguously in memory without padding. Requires custom attention masks but eliminates wasted computation on padding tokens.

### 18.12.3 GPU Utilization

Low GPU utilization is the primary source of cost waste:

- **Right-size instances:** If the model fits on 2 GPUs, do not use 4. Use GPU memory monitoring to determine the minimum instance type.
- **Model colocation:** Serve multiple small models on the same GPU using Triton's concurrent model execution.
- **Queue-based autoscaling:** Scale GPU instances based on request queue depth, not CPU utilization (which is misleading for GPU workloads).
- **Request batching at the API level:** Use an API gateway to collect requests and batch them before sending to the inference server.

### 18.12.4 Spot Instances and Auto-Scaling

For batch processing and non-latency-sensitive workloads, cloud spot/preemptible instances offer 60–80% cost savings:

```python
# Auto-scaling configuration (conceptual, Kubernetes HPA)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-server
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: request_queue_depth
      target:
        type: AverageValue
        averageValue: 5  # Scale up when avg queue depth > 5
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
      - type: Pods
        value: 2
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 min before scaling down
```

### 18.12.5 Cost Monitoring

Comprehensive cost monitoring should track:
- **Cost per request:** Total GPU-hours / total requests.
- **Cost per token:** Total GPU-hours / total tokens generated.
- **Cost per task completion:** For agent-style applications, total cost to complete a user's goal.
- **Idle cost:** GPU-hours where utilization < 10% — these represent pure waste.
- **Cost by model version:** Track cost changes when deploying new model versions or quantization levels.

```python
import time
from dataclasses import dataclass

@dataclass
class InferenceMetrics:
    request_id: str
    model: str
    input_tokens: int
    output_tokens: int
    ttft_ms: float
    total_latency_ms: float
    gpu_seconds: float

    @property
    def cost_usd(self) -> float:
        """Calculate cost based on GPU usage."""
        GPU_COST_PER_HOUR = 2.00  # A100 80GB
        return self.gpu_seconds * GPU_COST_PER_HOUR / 3600

    @property
    def cost_per_1k_tokens(self) -> float:
        """Cost per 1000 output tokens."""
        total_tokens = self.input_tokens + self.output_tokens
        return (self.cost_usd / total_tokens) * 1000 if total_tokens > 0 else 0
```

---

## Summary

LLM serving is a systems engineering challenge that requires deep understanding of hardware constraints, memory management, and concurrency. This chapter has covered the full stack: from foundational concepts (TTFT, TBT, throughput) through serving frameworks (BentoML, vLLM, TGI, Triton), optimization techniques (ONNX/TensorRT, quantization, speculative decoding), and advanced features (LoRA serving, structured generation, KV cache optimization).

The key insights are:

1. **PagedAttention** (vLLM) solved the KV cache memory fragmentation problem, enabling 2–4x throughput improvements.
2. **Continuous batching** keeps GPUs saturated by inserting and evicting requests at every decode step.
3. **Quantization** (GPTQ, AWQ, GGUF) makes large models accessible on smaller hardware with minimal quality loss.
4. **Speculative decoding** provides lossless 2–3x speedup by leveraging a small draft model.
5. **S-LoRA** enables multi-tenant serving with per-request adapter selection.
6. **Structured generation** (SGLang) guarantees valid output format through constrained decoding.
7. **Cost optimization** starts with model selection and extends through batching, caching, and infrastructure right-sizing.

The serving landscape evolves rapidly. New techniques — prefix-aware scheduling, disaggregated prefill and decode, cross-request KV cache sharing — continue to push the Pareto frontier of latency, throughput, and cost.

---

## Exercises

1. **vLLM deployment.** Deploy a Llama-3.1-8B model using vLLM with the OpenAI-compatible API. Benchmark throughput and latency at concurrency levels 1, 5, 10, 20, and 50. Plot the throughput-latency tradeoff curve.

2. **Quantization comparison.** Quantize Llama-3.1-8B with (a) GPTQ 4-bit, (b) AWQ 4-bit, and (c) GGUF Q4_K_M. Compare: model size, loading time, inference speed, and quality (using MMLU or a custom benchmark).

3. **Speculative decoding.** Implement speculative decoding from scratch using two Hugging Face models (a 7B target and a 1.3B draft). Measure the acceptance rate and speedup for different draft lengths $K = 1, 3, 5, 7$.

4. **Multi-LoRA serving.** Train 3 LoRA adapters for different tasks (summarization, translation, classification) and serve them simultaneously using vLLM's LoRA serving. Measure the overhead of adapter switching.

5. **Constrained generation.** Use SGLang to build a structured data extraction pipeline that converts unstructured text into JSON following a specific schema. Compare output validity rates with and without constrained decoding.

6. **Cost analysis.** For a production application serving 10,000 requests per day (average 200 input tokens, 500 output tokens), calculate the monthly cost for: (a) OpenAI GPT-4o API, (b) self-hosted Llama-3.1-70B on 4xA100, (c) self-hosted Llama-3.1-8B-AWQ on 1xA100. Include infrastructure, bandwidth, and engineering time estimates.

7. **End-to-end optimization.** Take a baseline Hugging Face inference pipeline and optimize it step by step: (a) add KV cache, (b) add continuous batching, (c) add quantization, (d) add prefix caching. Measure the improvement at each step.

---

## References

Chen, C., Borgeaud, S., Irving, G., Lespiau, J. B., Sifre, L., & Jumper, J. (2023). Accelerating large language model decoding with speculative sampling. *arXiv preprint arXiv:2302.01318*.

Dao, T. (2023). FlashAttention-2: Faster attention with better parallelism and work partitioning. *arXiv preprint arXiv:2307.08691*.

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. *Advances in Neural Information Processing Systems*, 35, 16344–16359.

Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2023). GPTQ: Accurate post-training quantization for generative pre-trained transformers. *Proceedings of the International Conference on Learning Representations*.

HuggingFace. (2023). Text generation inference. *GitHub Repository*. https://github.com/huggingface/text-generation-inference

Kirchenbauer, J., Geiping, J., Wen, Y., Katz, J., Miers, I., & Goldstein, T. (2023). A watermark for large language models. *Proceedings of the 40th International Conference on Machine Learning*, 17061–17084.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., ... & Stoica, I. (2023). Efficient memory management for large language model serving with PagedAttention. *Proceedings of the 29th Symposium on Operating Systems Principles*, 611–626.

Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast inference from transformers via speculative decoding. *Proceedings of the 40th International Conference on Machine Learning*, 19274–19286.

Lin, J., Tang, J., Tang, H., Yang, S., Chen, W. M., Wang, W. C., ... & Han, S. (2024). AWQ: Activation-aware weight quantization for on-device LLM compression and acceleration. *Proceedings of Machine Learning and Systems*, 6, 87–100.

NVIDIA. (2023). Triton Inference Server documentation. https://docs.nvidia.com/deeplearning/triton-inference-server/

Sheng, Y., Cao, S., Li, D., Hooper, C., Lee, N., Yang, S., ... & Stoica, I. (2023). S-LoRA: Serving thousands of concurrent LoRA adapters. *arXiv preprint arXiv:2311.03285*.

Yang, C., Sheng, S., Liu, Y., & others. (2024). BentoML: A framework for serving, managing, and deploying ML models. *GitHub Repository*. https://github.com/bentoml/BentoML

Zheng, L., Yin, L., Xie, Z., Huang, J., Sun, C., Yu, C. H., ... & Stoica, I. (2024). SGLang: Efficient execution of structured language model programs. *arXiv preprint arXiv:2312.07104*.
