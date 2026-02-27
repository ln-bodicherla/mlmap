# Chapter 6: Deep Learning with PyTorch

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Create, manipulate, and understand PyTorch tensors, including their memory layout, data types, and GPU acceleration.
2. Explain and leverage PyTorch's automatic differentiation engine (autograd), including computation graphs, gradient accumulation, and gradient control mechanisms.
3. Build neural network architectures using `nn.Module`, understanding parameter registration, forward computation, and model serialization.
4. Select and implement appropriate loss functions for regression, binary classification, and multi-class classification tasks, including custom losses.
5. Configure and use optimizers (SGD, Adam, AdamW) with learning rate schedulers for effective training.
6. Design custom `Dataset` and `DataLoader` pipelines for efficient data loading, including parallel workers and memory pinning.
7. Write complete training loops with mixed precision, gradient accumulation, early stopping, and checkpointing.
8. Apply regularization techniques (dropout, batch normalization, layer normalization, weight decay) and gradient clipping.
9. Use `torch.compile` for graph-level optimization and `torch.profiler` for performance analysis.
10. Understand the basics of custom CUDA extensions for performance-critical operations.

---

## 6.1 Tensors: The Foundation of PyTorch

At its core, PyTorch is a tensor computation library with automatic differentiation. A **tensor** is a multi-dimensional array --- a generalization of scalars (0D), vectors (1D), matrices (2D), and higher-order arrays. Tensors are the fundamental data structure through which all data flows in PyTorch (Paszke et al., 2019).

### 6.1.1 Tensor Creation

PyTorch provides numerous factory functions for creating tensors:

```python
import torch
import numpy as np

# From Python data
a = torch.tensor([1, 2, 3])                      # From list
b = torch.tensor([[1.0, 2.0], [3.0, 4.0]])       # 2D tensor

# From NumPy (shares memory --- zero-copy)
np_array = np.array([1.0, 2.0, 3.0])
c = torch.from_numpy(np_array)                    # Shares memory with np_array
np_array[0] = 99.0                                # Also changes c[0]

# Common factory functions
zeros = torch.zeros(3, 4)                         # 3x4 tensor of zeros
ones = torch.ones(2, 3, dtype=torch.float64)      # Specify dtype
rand = torch.rand(5, 5)                           # Uniform [0, 1)
randn = torch.randn(5, 5)                         # Standard normal
eye = torch.eye(4)                                # Identity matrix
arange = torch.arange(0, 10, 2)                   # [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, steps=5)          # [0, 0.25, 0.5, 0.75, 1.0]

# Like-functions (same shape, device, dtype as another tensor)
d = torch.zeros_like(b)
e = torch.randn_like(b)
```

### 6.1.2 Data Types

Tensor data types (dtypes) control precision, memory usage, and computation speed:

| dtype | Description | Size | Use Case |
|-------|-------------|------|----------|
| `torch.float32` | 32-bit float (default) | 4 bytes | General training |
| `torch.float64` | 64-bit float | 8 bytes | High-precision scientific computing |
| `torch.float16` | 16-bit float (half) | 2 bytes | Mixed precision training |
| `torch.bfloat16` | Brain float 16 | 2 bytes | Training on TPUs and modern GPUs |
| `torch.int64` | 64-bit integer (long) | 8 bytes | Indices, labels |
| `torch.int32` | 32-bit integer | 4 bytes | Indices |
| `torch.bool` | Boolean | 1 byte | Masks |

```python
# Casting
x = torch.randn(3, 3)
x_half = x.half()              # float32 -> float16
x_double = x.double()          # float32 -> float64
x_int = x.to(torch.int32)     # Truncates to integer
```

### 6.1.3 GPU Acceleration

PyTorch tensors can be moved to GPU for accelerated computation:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move tensor to GPU
x = torch.randn(1000, 1000, device=device)

# Move existing tensor
y = torch.randn(1000, 1000)
y = y.to(device)                # Returns a new tensor on the GPU
y = y.cuda()                    # Equivalent (if CUDA available)

# Check device
print(x.device)                 # cuda:0

# Move back to CPU (required for NumPy interop)
z = x.cpu().numpy()
```

GPU computation shines for large, parallel operations. For small tensors, the overhead of CPU-to-GPU transfer can exceed the computation savings.

### 6.1.4 Memory Layout and Contiguity

Tensors store data in a contiguous block of memory. Operations like `transpose()` and `permute()` change the **stride** (the step size in memory for each dimension) but do not rearrange the data in memory, creating a **non-contiguous** view:

```python
x = torch.randn(3, 4)
print(x.stride())           # (4, 1) --- row-major (C-contiguous)
print(x.is_contiguous())    # True

y = x.t()                   # Transpose --- shares memory, changes strides
print(y.stride())           # (1, 4) --- no longer C-contiguous
print(y.is_contiguous())    # False

# Some operations require contiguous memory
z = y.contiguous()           # Creates a new, contiguous copy
print(z.is_contiguous())    # True
```

Understanding contiguity matters for performance: non-contiguous memory access patterns lead to cache misses and slower computation. Many PyTorch operations require contiguous inputs and will call `.contiguous()` internally, incurring a hidden copy.

### 6.1.5 Essential Tensor Operations

```python
# Reshaping
x = torch.randn(2, 3, 4)
x.view(6, 4)                 # Reshape (requires contiguous memory)
x.reshape(6, 4)              # Reshape (works even if non-contiguous)
x.unsqueeze(0)               # Add dimension: (1, 2, 3, 4)
x.squeeze()                  # Remove dimensions of size 1
x.flatten(1)                 # Flatten from dim 1 onward: (2, 12)

# Indexing and slicing (same as NumPy)
x = torch.randn(5, 5)
x[0]                         # First row
x[:, 1]                      # Second column
x[x > 0]                     # Boolean indexing
x[1:3, 2:4]                  # Slice rows 1-2, columns 2-3

# Advanced indexing
indices = torch.tensor([0, 2, 4])
x[indices]                   # Gather specific rows
x.index_select(0, indices)   # Equivalent, explicit dimension

# Broadcasting
a = torch.randn(3, 1)
b = torch.randn(1, 4)
c = a + b                    # Shape: (3, 4)

# Matrix operations
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = A @ B                    # Matrix multiplication (3, 5)
C = torch.matmul(A, B)       # Equivalent
D = torch.bmm(                # Batch matrix multiply
    torch.randn(10, 3, 4),
    torch.randn(10, 4, 5)
)                             # Shape: (10, 3, 5)

# Einstein summation (powerful generalization)
E = torch.einsum('ij,jk->ik', A, B)    # Matrix multiply
F = torch.einsum('bij,bjk->bik',        # Batch matrix multiply
                  torch.randn(10, 3, 4),
                  torch.randn(10, 4, 5))

# Concatenation and stacking
x = torch.randn(3, 4)
y = torch.randn(3, 4)
torch.cat([x, y], dim=0)     # (6, 4) --- concatenate along rows
torch.stack([x, y], dim=0)   # (2, 3, 4) --- creates new dimension

# Reduction operations
x = torch.randn(3, 4)
x.sum()                      # Scalar sum of all elements
x.sum(dim=1)                 # Sum along columns: shape (3,)
x.mean(dim=0)                # Mean along rows: shape (4,)
x.max(dim=1)                 # Returns (values, indices) along dim 1
x.argmax(dim=1)              # Indices of maximum values along dim 1

# In-place operations (suffix with underscore)
x.add_(1)                    # In-place: x = x + 1
x.zero_()                    # Fill with zeros in-place
x.clamp_(min=0)              # ReLU in-place
```

**View vs. Reshape vs. Contiguous**: `view()` returns a tensor with the same underlying data but a different shape; it requires the tensor to be contiguous. `reshape()` works like `view()` when possible but creates a copy if necessary. Understanding when copies are made is important for both correctness (shared memory means modifications affect both tensors) and performance (copies consume memory and time).

---

## 6.2 Autograd: Automatic Differentiation

Autograd is PyTorch's automatic differentiation engine. It records operations on tensors into a **dynamic computation graph** (also called a **tape**) and uses this graph to compute gradients via backpropagation (reverse-mode automatic differentiation).

### 6.2.1 Computation Graphs

Every tensor operation creates a node in the computation graph. When `requires_grad=True`, PyTorch tracks all operations involving that tensor:

```python
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

z = x ** 2 + 3 * x * y + y ** 2   # z = x^2 + 3xy + y^2

# Compute gradients
z.backward()

print(x.grad)    # dz/dx = 2x + 3y = 2(2) + 3(3) = 13
print(y.grad)    # dz/dy = 3x + 2y = 3(2) + 2(3) = 12
```

The computation graph is **dynamic**: it is rebuilt from scratch on every forward pass. This means the graph can change at every iteration, enabling architectures with data-dependent control flow (loops, conditionals).

### 6.2.2 Key Autograd Mechanisms

**`requires_grad`**: Controls whether gradients are tracked for a tensor. Model parameters have `requires_grad=True` by default; data tensors do not.

**`backward()`**: Computes gradients by traversing the computation graph in reverse. By default, it computes gradients of a scalar output. For non-scalar outputs, a `gradient` argument (the vector in the vector-Jacobian product) must be provided.

**`grad`**: After `backward()`, the computed gradient is stored in `tensor.grad`. Gradients **accumulate** by default --- calling `backward()` multiple times adds gradients. This is why `zero_grad()` is called before each backward pass in training loops.

**`no_grad()`**: A context manager that disables gradient computation. Used during inference to save memory and computation:

```python
with torch.no_grad():
    predictions = model(X_test)    # No computation graph built
```

**`detach()`**: Returns a new tensor detached from the computation graph. The detached tensor shares the same storage but is treated as a constant:

```python
y = x.detach()    # y has same data as x but no gradient tracking
```

### 6.2.3 Gradient Accumulation

By default, `.grad` accumulates across `backward()` calls. This is useful for simulating larger batch sizes when GPU memory is limited:

```python
optimizer.zero_grad()
accumulation_steps = 4

for i, (inputs, targets) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()                 # Gradients accumulate

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()            # Update weights using accumulated gradients
        optimizer.zero_grad()       # Reset gradients
```

### 6.2.4 Custom Autograd Functions

For operations where autograd cannot automatically compute gradients (or for efficiency), you can define custom forward and backward passes:

```python
class CustomReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# Usage
output = CustomReLU.apply(input_tensor)
```

The `ctx` object is used to stash information for the backward pass via `save_for_backward`. The backward method must return one gradient per input to the forward method.

### 6.2.5 Higher-Order Gradients

For applications that require second-order derivatives (e.g., meta-learning, MAML, some regularization techniques like gradient penalty in WGANs), use `create_graph=True`:

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 3                          # y = x^3
grad1 = torch.autograd.grad(y, x, create_graph=True)[0]  # dy/dx = 3x^2 = 27
grad2 = torch.autograd.grad(grad1, x)[0]                  # d^2y/dx^2 = 6x = 18
```

The `create_graph=True` argument tells autograd to construct a computation graph for the gradient computation itself, enabling differentiation through the differentiation. This is more memory-intensive but necessary for second-order methods.

### 6.2.6 Common Autograd Pitfalls

Several common mistakes when working with autograd deserve explicit mention:

1. **Forgetting to call `zero_grad()`**: Gradients accumulate across backward calls. Without zeroing, each backward pass adds to the existing gradients, producing incorrect updates after the first iteration.

2. **In-place operations on tensors with `requires_grad=True`**: In-place operations (e.g., `x.add_(1)`) can overwrite values needed for gradient computation, causing errors or incorrect gradients. PyTorch detects and raises errors for most cases, but subtle bugs can occur.

3. **Moving between computation graph and NumPy**: Converting a tensor to NumPy (`.numpy()`) requires it to be on CPU and not require gradients. Use `.detach().cpu().numpy()` for safe conversion.

4. **Memory leaks from retaining computation graphs**: If you store loss values without calling `.item()`, you retain the entire computation graph in memory. Always use `loss.item()` for scalar values used in logging.

---

## 6.3 nn.Module: Building Neural Networks

`torch.nn.Module` is the base class for all neural network modules in PyTorch. It provides a clean abstraction for building, training, and deploying models.

### 6.3.1 Basic Module Structure

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNet(784, 256, 10)
print(model)
# SimpleNet(
#   (fc1): Linear(in_features=784, out_features=256, bias=True)
#   (relu): ReLU()
#   (fc2): Linear(in_features=256, out_features=10, bias=True)
# )
```

When a submodule (e.g., `nn.Linear`) is assigned as an attribute in `__init__`, it is automatically **registered**: its parameters appear in `model.parameters()`, it responds to `.to(device)`, and it is included in `model.state_dict()`.

### 6.3.2 Sequential, ModuleList, and ModuleDict

**`nn.Sequential`**: A container that chains modules in order:

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

**`nn.ModuleList`**: A list of modules that is properly registered (unlike a plain Python list):

```python
class DynamicNet(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1])
            for i in range(len(layer_sizes) - 1)
        ])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)
```

**`nn.ModuleDict`**: A dictionary of modules:

```python
class MultiHeadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(256, 128)
        self.heads = nn.ModuleDict({
            'classification': nn.Linear(128, 10),
            'regression': nn.Linear(128, 1)
        })

    def forward(self, x, task):
        x = torch.relu(self.shared(x))
        return self.heads[task](x)
```

### 6.3.3 Parameter Registration

Only tensors wrapped in `nn.Parameter` or assigned as submodules are registered:

```python
class CustomModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Registered parameter --- appears in model.parameters()
        self.weight = nn.Parameter(torch.randn(10, 5))
        # NOT registered --- plain tensor, will not be updated by optimizer
        self.constant = torch.tensor([1.0, 2.0, 3.0])

    def forward(self, x):
        return x @ self.weight + self.constant
```

To register a non-learnable tensor that should move with the model (e.g., to GPU), use `register_buffer`:

```python
self.register_buffer('running_mean', torch.zeros(10))
# self.running_mean is not in parameters() but is in state_dict()
# and moves with model.to(device)
```

### 6.3.4 Model Saving and Loading

```python
# Save entire model state
torch.save(model.state_dict(), 'model.pth')

# Load model state
model = SimpleNet(784, 256, 10)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Save complete checkpoint (model + optimizer + epoch)
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# Restore from checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

Always save `state_dict()` rather than the model object directly. The latter uses `pickle` and is fragile --- it breaks when code changes.

---

## 6.4 Loss Functions

The loss function quantifies the discrepancy between predictions and targets. Choosing the right loss is critical: it defines what the model optimizes.

### 6.4.1 Regression Losses

**Mean Squared Error (MSE)**:
$$\mathcal{L}_{\text{MSE}} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

Penalizes large errors heavily (quadratic). Sensitive to outliers.

```python
criterion = nn.MSELoss()
loss = criterion(predictions, targets)  # Both shape: (batch_size,)
```

**Huber Loss (Smooth L1)**:
$$\mathcal{L}_{\text{Huber}} = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\ \delta(|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$

Combines MSE for small errors with MAE for large errors. Robust to outliers.

```python
criterion = nn.HuberLoss(delta=1.0)
```

### 6.4.2 Classification Losses

**Cross-Entropy Loss (with logits)**:

For $K$-class classification, `nn.CrossEntropyLoss` combines `log_softmax` and `NLLLoss`:

$$\mathcal{L}_{\text{CE}} = -\sum_{k=1}^{K} y_k \log \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

For one-hot targets (most common), this simplifies to $-\log(\text{softmax}(z)_{y})$ where $y$ is the true class index.

```python
criterion = nn.CrossEntropyLoss()
# logits: (batch_size, num_classes) --- RAW scores, not softmax
# targets: (batch_size,) --- class indices (long integers)
loss = criterion(logits, targets)
```

Important: `CrossEntropyLoss` expects **raw logits**, not softmax outputs. Applying softmax before `CrossEntropyLoss` is a common bug that degrades both numerical stability and performance.

**Binary Cross-Entropy**:

For binary classification:

```python
# Option 1: BCE with explicit sigmoid
criterion = nn.BCELoss()
loss = criterion(torch.sigmoid(logits), targets.float())

# Option 2: BCE with logits (numerically stable --- preferred)
criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits, targets.float())
```

`BCEWithLogitsLoss` is preferred because it combines sigmoid and BCE in a numerically stable way using the log-sum-exp trick.

**Negative Log-Likelihood (NLL)**:

```python
criterion = nn.NLLLoss()
# Expects log-probabilities (output of log_softmax)
log_probs = torch.log_softmax(logits, dim=1)
loss = criterion(log_probs, targets)
```

### 6.4.3 Choosing the Right Loss Function

A common source of confusion is when to use which loss function. Here are the key guidelines:

| Task | Output Layer | Loss Function | Notes |
|------|-------------|---------------|-------|
| Regression | None (raw values) | `MSELoss` or `HuberLoss` | Huber for outlier robustness |
| Binary classification | None (raw logits) | `BCEWithLogitsLoss` | Do NOT apply sigmoid separately |
| Multi-class classification | None (raw logits) | `CrossEntropyLoss` | Do NOT apply softmax separately |
| Multi-label classification | None (raw logits) | `BCEWithLogitsLoss` | Each output is independent binary |
| Probabilistic (pre-softmax) | `log_softmax` | `NLLLoss` | Equivalent to CrossEntropyLoss |

The recurring theme is: let the loss function handle the activation internally. This is not just a convenience --- it is a numerical stability issue. Computing `log(softmax(x))` naively involves computing `exp(x)` which can overflow, then `log(...)` which can produce `-inf` for near-zero values. `CrossEntropyLoss` and `BCEWithLogitsLoss` use the log-sum-exp trick to avoid these issues.

### 6.4.4 Custom Loss Functions

Custom losses are regular Python functions that operate on tensors:

```python
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance (Lin et al., 2017).

    Down-weights well-classified examples, focusing training on hard examples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction='none'
        )
        probas = torch.sigmoid(logits)
        p_t = probas * targets + (1 - probas) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        loss = self.alpha * focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
```

---

## 6.5 Optimizers

Optimizers update model parameters to minimize the loss function. PyTorch provides a rich collection of optimizers in `torch.optim`.

### 6.5.1 Stochastic Gradient Descent (SGD)

The simplest optimizer updates parameters by subtracting the gradient scaled by the learning rate:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$

**SGD with Momentum** adds a velocity term that accumulates past gradients, smoothing updates and accelerating convergence in the relevant direction:

$$v_{t+1} = \mu v_t + \nabla_\theta \mathcal{L}(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta v_{t+1}$$

where $\mu$ is the momentum coefficient (typically 0.9).

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,    # L2 regularization
    nesterov=True          # Nesterov momentum (look-ahead)
)
```

### 6.5.2 Adam

Adam (Adaptive Moment Estimation) (Kingma & Ba, 2015) maintains per-parameter adaptive learning rates using estimates of the first and second moments of the gradients:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(first moment estimate)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(second moment estimate)}$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad \text{(bias correction)}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

Default hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

### 6.5.3 AdamW

AdamW (Loshchilov & Hutter, 2019) fixes a subtle issue in Adam: the standard implementation conflates L2 regularization with weight decay. In Adam, L2 regularization modifies the gradient, which interacts with the adaptive learning rate. AdamW applies weight decay directly to the parameters, decoupled from the gradient:

$$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

AdamW is now the default optimizer for most deep learning tasks, especially for training transformers.

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999)
)
```

### 6.5.4 Optimizer Internal State and Parameter Groups

Each optimizer maintains internal state (e.g., momentum buffers for SGD, moment estimates for Adam). This state is saved and restored via `optimizer.state_dict()`.

**Parameter groups** allow different hyperparameters for different parts of the model:

```python
optimizer = torch.optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 1e-5},    # Pretrained: low LR
    {'params': model.head.parameters(), 'lr': 1e-3}          # New layers: high LR
], weight_decay=0.01)
```

### 6.5.5 The Optimizer Step

The training loop interaction:

```python
optimizer.zero_grad()          # Clear accumulated gradients
loss = criterion(model(x), y)  # Forward pass
loss.backward()                # Backward pass: compute gradients
optimizer.step()               # Update parameters using gradients
```

`zero_grad()` is necessary because PyTorch accumulates gradients by default. Alternatively, `zero_grad(set_to_none=True)` sets gradients to `None` instead of zero, which is slightly more memory-efficient.

---

## 6.6 Custom Datasets and DataLoaders

Efficient data loading is crucial for training performance. PyTorch's data pipeline is built around two abstractions: `Dataset` and `DataLoader`.

### 6.6.1 The Dataset Class

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
```

The key contract:
- `__len__` returns the total number of samples.
- `__getitem__` returns a single sample (input, target) for a given index.

### 6.6.2 Image Dataset Example

```python
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, root_dir, labels_df, transform=None):
        self.root_dir = root_dir
        self.labels_df = labels_df
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = row['label']
        return image, label
```

### 6.6.3 IterableDataset

For data that does not fit in memory or arrives as a stream:

```python
from torch.utils.data import IterableDataset

class StreamDataset(IterableDataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            files = self.file_paths
        else:
            # Split files across workers
            per_worker = len(self.file_paths) // worker_info.num_workers
            start = worker_info.id * per_worker
            files = self.file_paths[start:start + per_worker]

        for path in files:
            with open(path) as f:
                for line in f:
                    yield process_line(line)
```

### 6.6.4 DataLoader

The `DataLoader` wraps a `Dataset` and provides batching, shuffling, parallel loading, and memory pinning:

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,           # Shuffle at every epoch
    num_workers=4,          # Parallel data loading processes
    pin_memory=True,        # Pin memory for faster GPU transfer
    drop_last=True,         # Drop incomplete last batch
    prefetch_factor=2       # Number of batches prefetched per worker
)

for batch_x, batch_y in train_loader:
    batch_x = batch_x.to(device, non_blocking=True)  # Async transfer
    batch_y = batch_y.to(device, non_blocking=True)
    # ... training step
```

**`num_workers`**: The number of subprocesses for data loading. Setting this to 0 loads data in the main process. A good starting point is the number of CPU cores divided by the number of GPUs. Too many workers can cause memory issues and CPU contention.

**`pin_memory`**: Allocates tensors in pinned (page-locked) memory, which enables faster and asynchronous transfer to GPU via `non_blocking=True`. Always use this when training on GPU.

**`collate_fn`**: Custom function to merge a list of samples into a batch. Useful when samples have variable lengths:

```python
def collate_variable_length(batch):
    """Pad sequences to the same length within a batch."""
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return padded, torch.tensor(labels), torch.tensor(lengths)

loader = DataLoader(dataset, batch_size=32, collate_fn=collate_variable_length)
```

---

## 6.7 Training Loops

The training loop is where all components come together. A well-structured training loop handles forward passes, loss computation, backpropagation, optimization, logging, validation, and checkpointing.

### 6.7.1 Basic Training Loop

```python
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()               # Set to training mode
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()                # Set to evaluation mode
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / total, correct / total
```

The distinction between `model.train()` and `model.eval()` is critical: it affects the behavior of layers like `Dropout` (active during training, disabled during evaluation) and `BatchNorm` (uses batch statistics during training, running statistics during evaluation).

### 6.7.2 Mixed Precision Training

Mixed precision training uses float16 for most computations while keeping a float32 master copy of the weights. This reduces memory usage by approximately 50% and speeds up training on modern GPUs with Tensor Cores (Micikevicius et al., 2018).

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, targets in train_loader:
    inputs = inputs.to(device)
    targets = targets.to(device)

    optimizer.zero_grad()

    # Forward pass in mixed precision
    with autocast(device_type='cuda'):
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**`autocast`** automatically selects the appropriate precision for each operation. Matrix multiplications and convolutions use float16; reductions and normalization use float32.

**`GradScaler`** scales the loss before backward to prevent float16 gradients from underflowing (becoming zero). It unscales gradients before the optimizer step and dynamically adjusts the scale factor.

### 6.7.3 Gradient Accumulation with Mixed Precision

```python
accumulation_steps = 4
scaler = GradScaler()

for i, (inputs, targets) in enumerate(train_loader):
    inputs = inputs.to(device)
    targets = targets.to(device)

    with autocast(device_type='cuda'):
        outputs = model(inputs)
        loss = criterion(outputs, targets) / accumulation_steps

    scaler.scale(loss).backward()

    if (i + 1) % accumulation_steps == 0:
        scaler.unscale_(optimizer)           # Unscale before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### 6.7.4 Early Stopping

Early stopping monitors validation performance and stops training when it stops improving:

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta
```

### 6.7.5 Checkpointing

Save model state periodically and upon best validation performance:

```python
best_val_loss = float('inf')

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader,
                                            criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, 'best_model.pth')
        print(f"Epoch {epoch}: New best val_loss = {val_loss:.4f}")
```

---

## 6.8 Regularization

Regularization techniques prevent overfitting by constraining model capacity. In deep learning, several complementary strategies are commonly used together.

### 6.8.1 Dropout

Dropout (Srivastava et al., 2014) randomly sets activations to zero during training with probability $p$. This prevents co-adaptation of neurons and acts as an implicit ensemble.

During training, outputs are scaled by $1/(1-p)$ so that expected values remain the same at test time (inverted dropout).

```python
class RegularizedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(p=0.5)      # Standard dropout
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)
```

**Variants**:
- **`nn.Dropout2d`** (spatial dropout): Drops entire feature maps. Used in CNNs.
- **`nn.AlphaDropout`**: Maintains self-normalizing properties. Used with SELU activation.

### 6.8.2 Batch Normalization

Batch Normalization (Ioffe & Szegedy, 2015) normalizes activations within each mini-batch, then applies a learned affine transformation:

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$

where $\mu_B$ and $\sigma_B^2$ are the batch mean and variance, and $\gamma, \beta$ are learnable scale and shift parameters.

Benefits: stabilizes training, allows higher learning rates, acts as a regularizer.

During evaluation, BatchNorm uses exponentially weighted running statistics computed during training.

```python
self.bn = nn.BatchNorm1d(128)    # For fully connected layers
self.bn2d = nn.BatchNorm2d(64)   # For convolutional layers (normalizes per channel)
```

### 6.8.3 Layer Normalization

Layer Normalization (Ba et al., 2016) normalizes across the feature dimension rather than the batch dimension:

$$\hat{x}_i = \frac{x_i - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}}$$

where $\mu_L$ and $\sigma_L^2$ are computed across all features for a single sample.

LayerNorm is independent of batch size and is the standard normalization in Transformers (Vaswani et al., 2017).

```python
self.ln = nn.LayerNorm(256)                    # For 1D features
self.ln2 = nn.LayerNorm([256, 16, 16])         # For spatial features
```

### 6.8.4 Group Normalization

Group Normalization (Wu & He, 2018) divides channels into groups and normalizes within each group. It bridges between LayerNorm (one group containing all channels) and InstanceNorm (each channel in its own group). GroupNorm is useful when batch sizes are too small for BatchNorm to work well.

```python
self.gn = nn.GroupNorm(num_groups=32, num_channels=256)
```

### 6.8.5 Weight Decay

Weight decay adds an L2 penalty on the parameters to the effective loss. In AdamW, weight decay is applied directly to the parameters after the gradient step (decoupled weight decay, as discussed in Section 6.5.3).

A subtlety: weight decay should typically **not** be applied to bias terms or normalization parameters. These have different roles than weight matrices and do not benefit from regularization. Applying weight decay to biases can even hurt performance. Most frameworks allow excluding specific parameters:

```python
# Separate parameters for weight decay
decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if 'bias' in name or 'norm' in name or 'bn' in name:
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer = torch.optim.AdamW([
    {'params': decay_params, 'weight_decay': 0.01},
    {'params': no_decay_params, 'weight_decay': 0.0}
], lr=1e-4)
```

### 6.8.6 Data Augmentation as Regularization

While not a model-level technique, data augmentation is one of the most effective forms of regularization. By applying random transformations to training data (random crops, flips, rotations, color jitter for images; synonym replacement, back-translation for text), we effectively increase the training set size and reduce overfitting. Modern techniques like Mixup (Zhang et al., 2018), CutMix, and RandAugment have become standard practice:

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25)
])
```

---

## 6.9 Learning Rate Scheduling

The learning rate is arguably the most important hyperparameter. Scheduling strategies adjust the learning rate during training to balance fast initial convergence with fine-grained late-stage optimization.

### 6.9.1 Step-Based Schedulers

```python
from torch.optim.lr_scheduler import (StepLR, ExponentialLR,
                                       CosineAnnealingLR, OneCycleLR,
                                       ReduceLROnPlateau)

# StepLR: Multiply LR by gamma every step_size epochs
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# ExponentialLR: Multiply LR by gamma every epoch
scheduler = ExponentialLR(optimizer, gamma=0.95)
```

### 6.9.2 Cosine Annealing

Cosine annealing smoothly decreases the learning rate following a cosine curve:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

```python
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
```

### 6.9.3 OneCycleLR

The One Cycle policy (Smith & Topin, 2019) uses a single cycle of learning rate that increases from a low value to a maximum, then decreases below the initial value, combined with inverse momentum scheduling:

```python
scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=100,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,            # Fraction of training for warmup
    anneal_strategy='cos'
)
# Note: OneCycleLR is stepped per batch, not per epoch
```

### 6.9.4 Warmup + Cosine Decay

A common strategy for transformer training: linearly warm up the learning rate for the first few steps, then cosine decay:

```python
from torch.optim.lr_scheduler import LambdaLR

def warmup_cosine_schedule(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer,
                     lr_lambda=lambda step: warmup_cosine_schedule(
                         step, warmup_steps=1000, total_steps=100000))
```

### 6.9.5 ReduceLROnPlateau

Reduces the learning rate when a monitored metric stops improving:

```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',          # Reduce when metric stops decreasing
    factor=0.1,          # Multiply LR by this factor
    patience=10,         # Wait this many epochs before reducing
    min_lr=1e-7
)

# In the training loop:
val_loss = evaluate(model, val_loader, criterion, device)
scheduler.step(val_loss)     # Pass the monitored metric
```

### 6.9.6 Integration with Training Loop

```python
# Per-epoch schedulers (StepLR, ExponentialLR, CosineAnnealingLR)
for epoch in range(num_epochs):
    train_one_epoch(...)
    scheduler.step()

# Per-batch schedulers (OneCycleLR, custom warmup)
for epoch in range(num_epochs):
    for batch in train_loader:
        train_step(...)
        scheduler.step()
```

---

## 6.10 Gradient Clipping

Gradient clipping prevents exploding gradients by limiting the magnitude of gradients during backpropagation. This is essential for training RNNs and beneficial for training very deep networks and transformers.

### 6.10.1 Clip by Norm

Scales the gradient vector so that its norm does not exceed a threshold:

$$\mathbf{g} \leftarrow \frac{\max\_norm}{\max(\|\mathbf{g}\|, \max\_norm)} \cdot \mathbf{g}$$

```python
# Clip the total norm of all gradients to max_norm
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

This scales all parameter gradients uniformly so that the total gradient norm is at most `max_norm`. The function returns the total gradient norm before clipping, which is useful for monitoring training stability.

### 6.10.2 Clip by Value

Clamps each gradient element independently:

```python
# Clip each gradient element to [-clip_value, clip_value]
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

Clip by norm is generally preferred because it preserves the direction of the gradient vector while only scaling its magnitude. Clip by value can distort the gradient direction.

### 6.10.3 Why Gradient Clipping Matters

Without gradient clipping, a single large gradient (from an outlier batch, a long sequence, or a loss spike) can catastrophically update all weights. Gradient clipping provides a safety net. Common `max_norm` values range from 0.5 to 5.0, with 1.0 being a standard choice for transformers.

---

## 6.11 torch.compile

Introduced in PyTorch 2.0, `torch.compile` is a function that takes a model and returns an optimized version. Under the hood, it uses **TorchDynamo** to capture the computation graph from Python bytecode and **TorchInductor** as the default backend to generate optimized GPU code (Ansel et al., 2024).

### 6.11.1 Basic Usage

```python
model = SimpleNet(784, 256, 10).to(device)

# Compile the model
compiled_model = torch.compile(model)

# Use exactly like the original model
output = compiled_model(input_tensor)
```

The first forward pass triggers compilation (a "warm-up" step), which can take tens of seconds. Subsequent calls use the compiled graph and are significantly faster.

### 6.11.2 Compilation Modes

```python
# Default mode: good balance of compile time and runtime performance
model = torch.compile(model, mode='default')

# Reduce-overhead: uses CUDA graphs to minimize kernel launch overhead
# Best for small models with many small operations
model = torch.compile(model, mode='reduce-overhead')

# Max-autotune: spends more time compiling to find the fastest implementation
# Tries multiple kernel implementations and selects the best
model = torch.compile(model, mode='max-autotune')
```

### 6.11.3 When to Use torch.compile

`torch.compile` is beneficial when:
- The model is computationally expensive (large batch sizes, deep models).
- The model has repetitive operations that benefit from kernel fusion.
- Training or inference runs long enough to amortize the compilation cost.

It may not help (or may hurt) when:
- The model has highly dynamic control flow (many graph breaks).
- Batch sizes are very small.
- The model is already I/O bound (data loading is the bottleneck).

```python
# Check for graph breaks (compatibility issues)
torch._dynamo.config.verbose = True
compiled_model = torch.compile(model, fullgraph=True)
# fullgraph=True will error if there are graph breaks, useful for debugging
```

---

## 6.12 torch.profiler

PyTorch's profiler provides detailed performance analysis of CPU and GPU operations, helping identify bottlenecks in training and inference.

### 6.12.1 Basic Profiling

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, (inputs, targets) in enumerate(train_loader):
        if step >= 5:    # Profile only a few steps
            break
        with record_function("forward"):
            outputs = model(inputs.to(device))
        with record_function("loss"):
            loss = criterion(outputs, targets.to(device))
        with record_function("backward"):
            loss.backward()
        with record_function("optimizer"):
            optimizer.step()
            optimizer.zero_grad()

# Print sorted by CUDA time
print(prof.key_averages().table(
    sort_by="cuda_time_total", row_limit=20
))
```

### 6.12.2 TensorBoard Integration

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, (inputs, targets) in enumerate(train_loader):
        if step >= 10:
            break
        outputs = model(inputs.to(device))
        loss = criterion(outputs, targets.to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        prof.step()       # Signal the profiler to move to the next step
```

The profiler output can be visualized in TensorBoard:

```bash
tensorboard --logdir=./log/profiler
```

### 6.12.3 Identifying Bottlenecks

Common bottlenecks revealed by profiling:

- **Data loading**: If CPU time dominates and GPU utilization is low, increase `num_workers` or use `pin_memory`.
- **CPU-GPU synchronization**: Operations like `.item()`, `print(tensor)`, or explicit `.cpu()` calls force synchronization. Minimize these in the inner loop.
- **Memory fragmentation**: Frequent allocation/deallocation of tensors. Use `torch.cuda.memory_summary()` to diagnose.
- **Small kernels**: Many small GPU operations have high launch overhead. `torch.compile` with `reduce-overhead` mode can help.

### 6.12.4 Memory Profiling

```python
# Track GPU memory usage
print(torch.cuda.memory_summary(device=device))

# Record memory snapshots for visualization
torch.cuda.memory._record_memory_history()
# ... run training ...
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
torch.cuda.memory._record_memory_history(enabled=None)
```

---

## 6.13 Distributed Training Basics

While a full treatment of distributed training appears in later chapters, understanding the basic APIs is essential for practical PyTorch usage.

### 6.13.1 DataParallel (DP)

The simplest form of multi-GPU training. It replicates the model on each GPU, splits the batch, computes forward/backward on each GPU, and aggregates gradients:

```python
model = nn.DataParallel(model)    # Wrap model
output = model(input)             # Automatically splits across GPUs
```

`DataParallel` has significant limitations: it uses a single-process multi-thread approach (limited by Python's GIL), has imbalanced GPU memory usage (GPU 0 collects all outputs), and is generally 20-30% slower than the alternative.

### 6.13.2 DistributedDataParallel (DDP)

The recommended approach for multi-GPU training. DDP spawns one process per GPU and uses NCCL for efficient all-reduce gradient synchronization:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group (typically called once at the start)
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])

# Use DistributedSampler to ensure each GPU gets different data
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
loader = DataLoader(dataset, sampler=sampler, batch_size=per_gpu_batch)
```

DDP is nearly linearly scalable across GPUs and is the foundation for large-scale training.

---

## 6.14 Custom CUDA Extensions

For performance-critical operations not available in PyTorch, you can write custom CUDA kernels and integrate them via `torch.utils.cpp_extension`.

### 6.13.1 C++/CUDA Extension Structure

A typical extension has three files:

**`my_op.cu`** --- the CUDA kernel:

```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel
__global__ void my_relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > 0.0f ? input[idx] : 0.0f;
    }
}

// C++ wrapper
torch::Tensor my_relu_cuda(torch::Tensor input) {
    auto output = torch::zeros_like(input);
    int size = input.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    my_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );

    return output;
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_relu", &my_relu_cuda, "Custom ReLU (CUDA)");
}
```

### 6.13.2 JIT Compilation

The simplest way to use a CUDA extension is JIT (Just-In-Time) compilation:

```python
from torch.utils.cpp_extension import load

my_op = load(
    name='my_op',
    sources=['my_op.cu'],
    verbose=True
)

# Use the custom operation
output = my_op.my_relu(input_tensor.cuda())
```

### 6.13.3 When to Write Custom CUDA Kernels

Custom CUDA extensions are warranted when:
- A fused operation (combining multiple steps into one kernel) provides significant speedup by reducing memory bandwidth.
- The operation is in the inner training loop and profiling shows it as a bottleneck.
- No existing PyTorch operation or `torch.compile` optimization achieves the desired performance.

For most practitioners, custom CUDA extensions are unnecessary. `torch.compile` with TorchInductor can automatically fuse many common operation patterns. Custom kernels are primarily the domain of library developers (FlashAttention, Triton kernels, etc.).

### 6.13.4 Triton: An Alternative to Raw CUDA

OpenAI Triton (Tillet et al., 2019) provides a Python-based language for writing GPU kernels that is significantly more accessible than raw CUDA. PyTorch's TorchInductor backend generates Triton kernels, and users can write custom Triton kernels directly:

```python
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.where(x > 0, x, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)
```

---

## 6.14 Putting It All Together: A Complete Training Script

The following script demonstrates all the components discussed in this chapter in a cohesive training pipeline:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
import time

# ---- Configuration ----
config = {
    'batch_size': 128,
    'epochs': 50,
    'lr': 1e-3,
    'weight_decay': 0.01,
    'grad_clip': 1.0,
    'patience': 10,
    'accumulation_steps': 1,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- Model ----
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

model = MLP(784, [512, 256, 128], 10).to(device)
model = torch.compile(model)    # Compile for performance

# ---- Data ----
# (Assume train_dataset and test_dataset are defined)
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=config['batch_size'],
                          shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_subset, batch_size=config['batch_size'] * 2,
                        num_workers=4, pin_memory=True)

# ---- Training Setup ----
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'],
                              weight_decay=config['weight_decay'])
scheduler = OneCycleLR(optimizer, max_lr=config['lr'],
                       epochs=config['epochs'],
                       steps_per_epoch=len(train_loader))
scaler = GradScaler()
early_stopping = EarlyStopping(patience=config['patience'], mode='min')

# ---- Training Loop ----
best_val_loss = float('inf')

for epoch in range(config['epochs']):
    # Training
    model.train()
    train_loss = 0
    start_time = time.time()

    for step, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(device_type='cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss = loss / config['accumulation_steps']

        scaler.scale(loss).backward()

        if (step + 1) % config['accumulation_steps'] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           config['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        train_loss += loss.item() * config['accumulation_steps']

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    val_loss /= len(val_loader)
    val_acc = correct / total
    elapsed = time.time() - start_time

    print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
          f"Time: {elapsed:.1f}s | LR: {scheduler.get_last_lr()[0]:.2e}")

    # Checkpointing
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')

    # Early stopping
    early_stopping(val_loss)
    if early_stopping.should_stop:
        print(f"Early stopping at epoch {epoch+1}")
        break
```

---

## Exercises

1. **Tensor Operations**: Create a function that performs batch matrix multiplication without using `torch.bmm` or `@`. Use only basic operations like `unsqueeze`, `expand`, and `sum`. Verify it gives the same result as `torch.bmm`.

2. **Autograd Investigation**: Write a function $f(x) = x^3 \sin(x)$ and compute $f'(x)$, $f''(x)$, and $f'''(x)$ at $x = \pi$ using `torch.autograd.grad` with `create_graph=True`. Verify against the analytical derivatives.

3. **Custom Module**: Implement a `ResidualBlock` module with two linear layers, batch normalization, GELU activation, and a skip connection. Then build a model with 5 stacked residual blocks.

4. **Loss Functions**: Implement label-smoothed cross-entropy loss. Given a smoothing factor $\epsilon = 0.1$ and $K$ classes, the target becomes $(1-\epsilon)\mathbf{y}_{\text{one-hot}} + \epsilon/K$. Compare training with and without label smoothing on CIFAR-10.

5. **Data Pipeline**: Create a custom `Dataset` for a CSV file containing image paths and labels. Implement data augmentation (random crop, horizontal flip, color jitter) using `torchvision.transforms`. Benchmark `DataLoader` performance with different `num_workers` values.

6. **Mixed Precision**: Train the same model with full precision (float32) and mixed precision. Compare (a) training time per epoch, (b) peak GPU memory usage, and (c) final validation accuracy. Use `torch.cuda.max_memory_allocated()` for memory measurement.

7. **Learning Rate Scheduling**: Implement a custom learning rate scheduler that combines linear warmup (first 10% of training) with cosine annealing (remaining 90%). Plot the learning rate over 100 epochs for a base LR of 1e-3.

8. **Profiling**: Profile a CNN training loop on CIFAR-10 using `torch.profiler`. Identify the top 5 most time-consuming operations. Try `torch.compile` and re-profile. Quantify the speedup.

9. **Gradient Clipping Analysis**: Train an LSTM on a sequence prediction task with and without gradient clipping. Monitor the gradient norm over training. At what gradient norm threshold does training become unstable without clipping?

10. **Complete Pipeline**: Build a complete training pipeline for MNIST digit classification that includes: custom Dataset, DataLoader with `pin_memory`, an MLP with dropout and layer normalization, AdamW optimizer, cosine annealing schedule with warmup, mixed precision training, gradient clipping, early stopping, and checkpointing. Target > 98.5% test accuracy.

---

## References

Ansel, J., Yang, E., He, H., et al. (2024). PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation. *Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems*, 929--947.

Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization. *arXiv preprint arXiv:1607.06450*.

Howard, J., & Gugger, S. (2020). *Deep Learning for Coders with fastai and PyTorch*. O'Reilly Media.

Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *Proceedings of the 32nd International Conference on Machine Learning*, 448--456.

Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *Proceedings of the 3rd International Conference on Learning Representations*.

Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). Focal Loss for Dense Object Detection. *Proceedings of the IEEE International Conference on Computer Vision*, 2980--2988.

Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *Proceedings of the 7th International Conference on Learning Representations*.

Micikevicius, P., Narang, S., Alben, J., et al. (2018). Mixed Precision Training. *Proceedings of the 6th International Conference on Learning Representations*.

Paszke, A., Gross, S., Massa, F., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Advances in Neural Information Processing Systems*, 32, 8024--8035.

Smith, L. N., & Topin, N. (2019). Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates. *Artificial Intelligence and Machine Learning for Multi-Domain Operations Applications*, 11006, 369--386.

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *Journal of Machine Learning Research*, 15(56), 1929--1958.

Tillet, P., Kung, H. T., & Cox, D. (2019). Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations. *Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages*, 10--19.

Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems*, 30, 5998--6008.

Wu, Y., & He, K. (2018). Group Normalization. *Proceedings of the European Conference on Computer Vision*, 3--19.
