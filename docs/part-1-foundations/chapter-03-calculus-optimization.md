# Chapter 3: Calculus and Optimization

> *"Learning is optimization. Every model that improves from data does so because an optimization algorithm descends a loss landscape shaped by calculus."*

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Compute derivatives, partial derivatives, and gradients, and interpret them geometrically as slopes and directions of steepest ascent.
2. Apply the chain rule to computation graphs, understanding backpropagation as reverse-mode automatic differentiation.
3. Implement backpropagation from scratch for a simple computational graph (micrograd-style).
4. Derive and implement gradient descent variants: vanilla, mini-batch, and stochastic.
5. Derive the update rules for SGD with momentum, RMSProp, Adam, and AdamW, understanding the motivation for each.
6. Implement and compare learning rate schedules: step decay, exponential, cosine annealing, warmup + cosine, and OneCycleLR.
7. Compute Jacobians and Hessians and explain their role in second-order optimization methods.
8. Distinguish convex from non-convex optimization and reason about saddle points and local minima in deep learning.
9. Apply numerical stability techniques: the log-sum-exp trick, stable softmax, and gradient clipping.
10. Formulate and solve constrained optimization problems using Lagrange multipliers and KKT conditions.

---

## 3.1 Derivatives and the Chain Rule

### 3.1.1 The Derivative as Rate of Change

The derivative of a function $f: \mathbb{R} \to \mathbb{R}$ at a point $x$ is defined as:

$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

Geometrically, $f'(x)$ is the slope of the tangent line to $f$ at $x$. In machine learning, $f$ is typically a loss function, $x$ represents model parameters, and $f'(x)$ tells us how the loss changes as we change the parameters.

**Key derivatives for ML:**

| Function | Derivative | Where it appears |
|---|---|---|
| $f(x) = x^n$ | $nx^{n-1}$ | Polynomial features |
| $f(x) = e^x$ | $e^x$ | Softmax, exponential family |
| $f(x) = \ln x$ | $1/x$ | Log-likelihood, cross-entropy |
| $f(x) = \sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ | Sigmoid activation, logistic regression |
| $f(x) = \tanh(x)$ | $1 - \tanh^2(x)$ | Tanh activation |
| $f(x) = \max(0, x)$ | $\begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases}$ | ReLU activation |

### 3.1.2 The Chain Rule

The chain rule is the single most important calculus result for machine learning. If $y = f(g(x))$, then:

$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx} = f'(g(x)) \cdot g'(x)$$

For a composition of multiple functions $y = f_n(f_{n-1}(\cdots f_1(x)))$:

$$\frac{dy}{dx} = f_n'(z_{n-1}) \cdot f_{n-1}'(z_{n-2}) \cdots f_1'(x)$$

where $z_i = f_i(z_{i-1})$ and $z_0 = x$.

This is exactly what happens in a neural network: data passes through layers (functions), and the chain rule tells us how the loss depends on each layer's parameters.

### 3.1.3 Computation Graphs

A **computation graph** makes the chain rule visual and algorithmic. Every mathematical expression can be decomposed into elementary operations, forming a directed acyclic graph (DAG).

Consider the simple function $L = (wx + b - y)^2$:

```
x ---\
      (*) --> z1 = wx
w ---/           \
                  (+) --> z2 = wx + b
b ---------------/           \
                              (-) --> z3 = wx + b - y
y ---------------------------/           \
                                          (^2) --> L = z3^2
```

**Forward pass** (compute $L$ from inputs):
1. $z_1 = w \cdot x$
2. $z_2 = z_1 + b$
3. $z_3 = z_2 - y$
4. $L = z_3^2$

**Backward pass** (compute $\frac{\partial L}{\partial w}$ and $\frac{\partial L}{\partial b}$ using the chain rule):
1. $\frac{\partial L}{\partial z_3} = 2z_3$
2. $\frac{\partial L}{\partial z_2} = \frac{\partial L}{\partial z_3} \cdot \frac{\partial z_3}{\partial z_2} = 2z_3 \cdot 1 = 2z_3$
3. $\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial z_2} \cdot 1 = 2z_3$
4. $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z_1} \cdot \frac{\partial z_1}{\partial w} = 2z_3 \cdot x = 2(wx + b - y) \cdot x$
5. $\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z_2} \cdot \frac{\partial z_2}{\partial b} = 2z_3 \cdot 1 = 2(wx + b - y)$

```python
# Numerical verification
w, x, b, y = 2.0, 3.0, 1.0, 5.0

# Forward pass
z1 = w * x        # 6.0
z2 = z1 + b       # 7.0
z3 = z2 - y       # 2.0
L = z3 ** 2       # 4.0

# Backward pass (analytical)
dL_dz3 = 2 * z3         # 4.0
dL_dz2 = dL_dz3 * 1     # 4.0
dL_dz1 = dL_dz2 * 1     # 4.0
dL_dw = dL_dz1 * x      # 12.0
dL_db = dL_dz2 * 1      # 4.0

# Numerical verification (finite differences)
eps = 1e-5
dL_dw_numerical = ((((w+eps)*x + b - y)**2) - L) / eps  # ≈ 12.0
print(f"Analytical: {dL_dw}, Numerical: {dL_dw_numerical:.6f}")
```

---

## 3.2 Partial Derivatives and Gradients

### 3.2.1 Partial Derivatives

When a function has multiple inputs, $f: \mathbb{R}^n \to \mathbb{R}$, the **partial derivative** $\frac{\partial f}{\partial x_i}$ measures the rate of change of $f$ with respect to $x_i$ while holding all other variables constant.

Example: For $f(x_1, x_2) = x_1^2 + 3x_1 x_2 + x_2^2$:

$$\frac{\partial f}{\partial x_1} = 2x_1 + 3x_2, \quad \frac{\partial f}{\partial x_2} = 3x_1 + 2x_2$$

### 3.2.2 The Gradient

The **gradient** $\nabla f$ collects all partial derivatives into a vector:

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

**Geometric interpretation**: The gradient $\nabla f(\mathbf{x})$ points in the direction of steepest ascent of $f$ at $\mathbf{x}$. Its magnitude $\|\nabla f(\mathbf{x})\|$ is the rate of increase in that direction.

This is why gradient descent works: to minimize $f$, we move in the direction opposite to the gradient.

### 3.2.3 Gradient of Common ML Loss Functions

**Mean Squared Error (MSE):**

$$L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \frac{1}{n}\|\mathbf{y} - \hat{\mathbf{y}}\|^2$$

$$\nabla_{\hat{\mathbf{y}}} L = -\frac{2}{n}(\mathbf{y} - \hat{\mathbf{y}})$$

**Cross-Entropy Loss (for binary classification):**

$$L = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \ln(\hat{y}_i) + (1-y_i)\ln(1-\hat{y}_i)\right]$$

$$\frac{\partial L}{\partial \hat{y}_i} = -\frac{1}{n}\left[\frac{y_i}{\hat{y}_i} - \frac{1 - y_i}{1 - \hat{y}_i}\right]$$

When combined with the sigmoid, the gradient simplifies beautifully:

$$\frac{\partial L}{\partial z_i} = \frac{1}{n}(\sigma(z_i) - y_i)$$

This is why logistic regression uses cross-entropy with sigmoid — the gradient is simply the prediction error.

---

## 3.3 Backpropagation

Backpropagation is the algorithm that makes training deep neural networks computationally feasible. It is nothing more than the **chain rule applied systematically to a computation graph**, but applied in a specific order (reverse) that is maximally efficient.

### 3.3.1 Forward Mode vs. Reverse Mode Automatic Differentiation

There are two ways to apply the chain rule to a computation graph:

**Forward mode**: Start from inputs, propagate derivatives forward. To compute $\frac{\partial L}{\partial w_i}$, we need one forward pass per parameter. Cost: $O(n)$ forward passes for $n$ parameters.

**Reverse mode** (backpropagation): Start from the output, propagate derivatives backward. A single backward pass computes $\frac{\partial L}{\partial w_i}$ for *all* parameters simultaneously. Cost: $O(1)$ backward passes.

Since neural networks have millions (or billions) of parameters but a single scalar loss, reverse mode is overwhelmingly more efficient. This is backpropagation.

### 3.3.2 Backpropagation for a Two-Layer Network

Consider a two-layer network:

$$\hat{\mathbf{y}} = \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$$

$$L = \frac{1}{2}\|\mathbf{y} - \hat{\mathbf{y}}\|^2$$

**Forward pass:**
1. $\mathbf{z}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1$
2. $\mathbf{a}_1 = \text{ReLU}(\mathbf{z}_1)$
3. $\mathbf{z}_2 = \mathbf{W}_2 \mathbf{a}_1 + \mathbf{b}_2$
4. $L = \frac{1}{2}\|\mathbf{y} - \mathbf{z}_2\|^2$

**Backward pass:**
1. $\frac{\partial L}{\partial \mathbf{z}_2} = -(\mathbf{y} - \mathbf{z}_2) = \hat{\mathbf{y}} - \mathbf{y}$
2. $\frac{\partial L}{\partial \mathbf{W}_2} = \frac{\partial L}{\partial \mathbf{z}_2} \mathbf{a}_1^T$
3. $\frac{\partial L}{\partial \mathbf{b}_2} = \frac{\partial L}{\partial \mathbf{z}_2}$
4. $\frac{\partial L}{\partial \mathbf{a}_1} = \mathbf{W}_2^T \frac{\partial L}{\partial \mathbf{z}_2}$
5. $\frac{\partial L}{\partial \mathbf{z}_1} = \frac{\partial L}{\partial \mathbf{a}_1} \odot \mathbb{1}[\mathbf{z}_1 > 0]$ (element-wise, ReLU derivative)
6. $\frac{\partial L}{\partial \mathbf{W}_1} = \frac{\partial L}{\partial \mathbf{z}_1} \mathbf{x}^T$
7. $\frac{\partial L}{\partial \mathbf{b}_1} = \frac{\partial L}{\partial \mathbf{z}_1}$

### 3.3.3 Building Backpropagation from Scratch (Micrograd)

Following Karpathy's micrograd approach (Karpathy, 2022), we build a tiny autograd engine that supports scalar-valued computation graphs:

```python
import math

class Value:
    """A scalar value with automatic differentiation support."""

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad       # d(a+b)/da = 1
            other.grad += out.grad      # d(a+b)/db = 1
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad  # d(ab)/da = b
            other.grad += self.data * out.grad  # d(ab)/db = a
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad  # d(e^x)/dx = e^x
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1.0 / self.data) * out.grad  # d(ln x)/dx = 1/x
        out._backward = _backward
        return out

    def backward(self):
        """Topological sort + reverse-order backward pass."""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0  # dL/dL = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other


# Demonstration: training a tiny network
# Input: (x1, x2), Target: y
x1 = Value(2.0)
x2 = Value(3.0)
y = Value(1.0)

# Parameters
w1 = Value(0.5)
w2 = Value(-0.3)
b = Value(0.1)

# Forward pass: simple linear model
pred = (w1 * x1 + w2 * x2 + b)
loss = (pred - y) ** 2

print(f"Prediction: {pred.data:.4f}")
print(f"Loss: {loss.data:.4f}")

# Backward pass
loss.backward()

print(f"dL/dw1 = {w1.grad:.4f}")
print(f"dL/dw2 = {w2.grad:.4f}")
print(f"dL/db  = {b.grad:.4f}")

# Gradient descent step
lr = 0.01
w1.data -= lr * w1.grad
w2.data -= lr * w2.grad
b.data -= lr * b.grad
```

---

## 3.4 Gradient Descent

Gradient descent is the workhorse optimization algorithm of machine learning. It iteratively adjusts parameters to minimize the loss function by moving in the direction opposite to the gradient.

### 3.4.1 Vanilla (Batch) Gradient Descent

Given a loss function $\mathcal{L}(\boldsymbol{\theta})$ over the entire training set, the update rule is:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t)$$

where $\eta$ is the learning rate.

**Properties:**
- Uses the *entire* training set to compute the gradient.
- Converges smoothly for convex problems.
- Impractical for large datasets (must process all $n$ samples before one update).

### 3.4.2 Stochastic Gradient Descent (SGD)

Use a single random sample to estimate the gradient:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla_{\boldsymbol{\theta}} \ell(\boldsymbol{\theta}_t; x_i, y_i)$$

**Properties:**
- Noisy gradient estimate, but unbiased: $\mathbb{E}[\nabla \ell_i] = \nabla \mathcal{L}$.
- The noise can help escape local minima.
- High variance leads to oscillation.

### 3.4.3 Mini-Batch Gradient Descent

The practical compromise: use a random mini-batch of $B$ samples:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \frac{1}{B} \sum_{i \in \mathcal{B}_t} \nabla_{\boldsymbol{\theta}} \ell(\boldsymbol{\theta}_t; x_i, y_i)$$

**Properties:**
- Lower variance than SGD (by a factor of $\sqrt{B}$).
- Efficient: leverages GPU parallelism.
- Typical batch sizes: 32, 64, 128, 256, 512.

```python
def mini_batch_sgd(model, X, y, lr=0.01, batch_size=32, epochs=100):
    """Mini-batch gradient descent with NumPy."""
    n = len(X)
    losses = []

    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Forward pass
            predictions = model.forward(X_batch)
            loss = np.mean((predictions - y_batch) ** 2)
            epoch_loss += loss * len(X_batch)

            # Backward pass (compute gradients)
            grad = model.compute_gradients(X_batch, y_batch)

            # Update parameters
            for param, g in zip(model.parameters(), grad):
                param -= lr * g

        losses.append(epoch_loss / n)
    return losses
```

---

## 3.5 Advanced Optimizers

Vanilla SGD has well-known limitations: it oscillates in ravines (directions with high curvature), progresses slowly along shallow dimensions, and requires careful learning rate tuning. Advanced optimizers address these issues by adapting the effective learning rate per parameter.

### 3.5.1 SGD with Momentum

**Motivation:** In a loss landscape with a long narrow valley, the gradient oscillates across the valley (high curvature) while making slow progress along it (low curvature). Momentum dampens oscillations and accelerates progress in consistent directions.

**Derivation:** Introduce a velocity vector $\mathbf{v}$ that accumulates past gradients with exponential decay:

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t)$$
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \mathbf{v}_t$$

The velocity $\mathbf{v}_t$ is an exponential moving average of gradients. The hyperparameter $\beta$ (typically 0.9) controls how much history is retained. In directions where gradients consistently point the same way, momentum builds up; where they oscillate, momentum cancels out.

**Nesterov momentum** (Nesterov, 1983) looks ahead before computing the gradient:

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t - \eta \beta \mathbf{v}_{t-1})$$
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \mathbf{v}_t$$

```python
class SGDMomentum:
    def __init__(self, params, lr=0.01, momentum=0.9, nesterov=False):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocities = [np.zeros_like(p) for p in self.params]

    def step(self, grads):
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            self.velocities[i] = self.momentum * self.velocities[i] + grad
            if self.nesterov:
                param -= self.lr * (grad + self.momentum * self.velocities[i])
            else:
                param -= self.lr * self.velocities[i]
```

### 3.5.2 RMSProp

**Motivation:** Different parameters may need different learning rates. Parameters associated with frequently occurring features should have smaller learning rates (they have already learned a lot), while rare features need larger rates. RMSProp (Hinton, 2012) adapts the learning rate per parameter using the magnitude of recent gradients.

**Derivation:** Maintain an exponential moving average of squared gradients:

$$\mathbf{s}_t = \rho \mathbf{s}_{t-1} + (1 - \rho) \mathbf{g}_t^2$$
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\mathbf{s}_t} + \epsilon} \mathbf{g}_t$$

where $\mathbf{g}_t = \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t)$, $\rho = 0.99$, and $\epsilon = 10^{-8}$ prevents division by zero.

The denominator $\sqrt{\mathbf{s}_t}$ normalizes each parameter's gradient by its recent RMS (root mean square). Parameters with large gradients get their effective learning rate reduced; parameters with small gradients get it increased.

```python
class RMSProp:
    def __init__(self, params, lr=0.001, rho=0.99, epsilon=1e-8):
        self.params = list(params)
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon
        self.sq_avg = [np.zeros_like(p) for p in self.params]

    def step(self, grads):
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            self.sq_avg[i] = self.rho * self.sq_avg[i] + (1 - self.rho) * grad**2
            param -= self.lr * grad / (np.sqrt(self.sq_avg[i]) + self.epsilon)
```

### 3.5.3 Adam (Adaptive Moment Estimation)

**Motivation:** Adam (Kingma & Ba, 2015) combines the best of momentum (first moment) and RMSProp (second moment), with bias correction to account for initialization at zero.

**Derivation:**

First moment estimate (mean of gradients, like momentum):
$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t$$

Second moment estimate (mean of squared gradients, like RMSProp):
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2$$

**Bias correction:** Since $\mathbf{m}_0 = \mathbf{v}_0 = \mathbf{0}$, the estimates are biased toward zero in early steps. We can show (by expanding the recursion) that:

$$\mathbb{E}[\mathbf{m}_t] = (1 - \beta_1^t) \mathbb{E}[\mathbf{g}_t]$$

So the bias-corrected estimates are:

$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$

**Update rule:**
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t$$

Default hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 0.001$.

```python
class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros_like(p) for p in self.params]  # First moment
        self.v = [np.zeros_like(p) for p in self.params]  # Second moment
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # Update biased moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
```

### 3.5.4 AdamW (Adam with Decoupled Weight Decay)

**Motivation:** Loshchilov and Hutter (2019) showed that the standard way of combining L2 regularization with Adam is incorrect. In standard Adam, the L2 penalty gradient $\lambda \boldsymbol{\theta}$ is added to the gradient before the adaptive scaling, which effectively reduces the regularization for parameters with large gradients. AdamW fixes this by applying weight decay directly to the parameters, *outside* of the adaptive mechanism.

**The difference:**

*Adam with L2 regularization (incorrect coupling):*
$$\mathbf{g}_t = \nabla \mathcal{L}_t + \lambda \boldsymbol{\theta}_t$$
$$\text{... then standard Adam update with } \mathbf{g}_t$$

*AdamW (decoupled weight decay):*
$$\mathbf{g}_t = \nabla \mathcal{L}_t \quad \text{(no regularization in gradient)}$$
$$\text{... standard Adam update with } \mathbf{g}_t$$
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_{t+1} - \eta \lambda \boldsymbol{\theta}_t \quad \text{(weight decay applied separately)}$$

```python
class AdamW:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # Update moments (using only the task gradient, not regularization)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Adam update
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # Decoupled weight decay (applied SEPARATELY)
            param -= self.lr * self.weight_decay * param
```

AdamW is the default optimizer for training transformers, including BERT, GPT, LLaMA, and virtually all modern large language models.

---

## 3.6 Learning Rate Scheduling

The learning rate is the most important hyperparameter. Too large and training diverges; too small and it converges slowly or gets stuck. Learning rate schedules adjust $\eta$ over time, typically starting high (for fast initial progress) and decreasing (for fine-grained convergence).

### 3.6.1 Step Decay

Reduce the learning rate by a factor $\gamma$ every $k$ epochs:

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / k \rfloor}$$

```python
def step_decay(epoch, lr_init=0.1, drop_factor=0.1, drop_every=30):
    return lr_init * (drop_factor ** (epoch // drop_every))
```

### 3.6.2 Exponential Decay

$$\eta_t = \eta_0 \cdot \gamma^t$$

### 3.6.3 Cosine Annealing

Smoothly decrease the learning rate following a cosine curve (Loshchilov & Hutter, 2017):

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

### 3.6.4 Warmup + Cosine Decay

The standard schedule for transformer training. Start with a linear warmup from 0 to $\eta_{\max}$ over $T_w$ steps, then cosine decay:

$$\eta_t = \begin{cases} \eta_{\max} \cdot \frac{t}{T_w} & t \leq T_w \\ \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t - T_w}{T - T_w}\pi\right)\right) & t > T_w \end{cases}$$

```python
def warmup_cosine_schedule(step, total_steps, warmup_steps,
                            lr_max=1e-3, lr_min=1e-5):
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))

# Visualization
import matplotlib.pyplot as plt
steps = range(10000)
lrs = [warmup_cosine_schedule(s, 10000, 1000) for s in steps]
plt.plot(steps, lrs)
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.title("Warmup + Cosine Decay Schedule")
```

### 3.6.5 OneCycleLR

The 1cycle policy (Smith & Topin, 2019) starts at a low learning rate, increases to a maximum (warmup phase), then decreases back to below the initial rate (annealing phase). The key insight is that a high learning rate acts as regularization.

```python
# PyTorch implementation
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    total_steps=total_steps,
    pct_start=0.3,          # 30% of training is warmup
    anneal_strategy='cos',
    div_factor=25,           # initial_lr = max_lr / 25
    final_div_factor=1e4,    # final_lr = initial_lr / 1e4
)

for batch in dataloader:
    loss = train_step(batch)
    loss.backward()
    optimizer.step()
    scheduler.step()  # Update LR every step
```

---

## 3.7 Jacobians and Hessians

### 3.7.1 The Jacobian Matrix

For a vector-valued function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the **Jacobian** $\mathbf{J} \in \mathbb{R}^{m \times n}$ is the matrix of all partial derivatives:

$$J_{ij} = \frac{\partial f_i}{\partial x_j}$$

The Jacobian generalizes the gradient (which is the Jacobian of a scalar-valued function).

**Example:** The softmax function $\sigma: \mathbb{R}^n \to \mathbb{R}^n$ where $\sigma_i(\mathbf{z}) = \frac{e^{z_i}}{\sum_j e^{z_j}}$ has Jacobian:

$$J_{ij} = \frac{\partial \sigma_i}{\partial z_j} = \begin{cases} \sigma_i (1 - \sigma_j) & i = j \\ -\sigma_i \sigma_j & i \neq j \end{cases} = \sigma_i(\delta_{ij} - \sigma_j)$$

This can be written compactly as: $\mathbf{J} = \text{diag}(\boldsymbol{\sigma}) - \boldsymbol{\sigma}\boldsymbol{\sigma}^T$.

### 3.7.2 The Hessian Matrix

For a scalar-valued function $f: \mathbb{R}^n \to \mathbb{R}$, the **Hessian** $\mathbf{H} \in \mathbb{R}^{n \times n}$ is the matrix of second-order partial derivatives:

$$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

The Hessian encodes the curvature of the function:
- If $\mathbf{H}$ is positive definite at a critical point, it is a local minimum.
- If $\mathbf{H}$ is negative definite, it is a local maximum.
- If $\mathbf{H}$ has both positive and negative eigenvalues, it is a saddle point.

### 3.7.3 Second-Order Optimization

**Newton's method** uses the Hessian to take curvature into account:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \mathbf{H}^{-1} \nabla f(\boldsymbol{\theta}_t)$$

This converges much faster than gradient descent (quadratically near the optimum vs. linearly), but computing and inverting the Hessian is $O(n^2)$ storage and $O(n^3)$ computation — infeasible for neural networks with millions of parameters.

**L-BFGS** (Limited-memory BFGS) approximates the inverse Hessian using a low-rank update from the last $m$ gradient evaluations (typically $m = 10$-$20$). It is practical for moderate-dimensional problems:

```python
from scipy.optimize import minimize

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    return np.array([
        -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2),
        200 * (x[1] - x[0]**2)
    ])

result = minimize(rosenbrock, x0=[0, 0], jac=rosenbrock_grad, method='L-BFGS-B')
print(f"Minimum at: {result.x}")   # [1.0, 1.0]
print(f"Iterations: {result.nit}")  # ~30 (vs. thousands for gradient descent)
```

**Practical note:** Second-order methods are rarely used in deep learning due to cost, but they appear in:
- Classical ML (logistic regression, CRFs): `scipy.optimize.minimize` with L-BFGS
- Neural architecture search: natural gradient methods
- Fisher information matrix: used in elastic weight consolidation (continual learning)

---

## 3.8 Convex vs. Non-Convex Optimization

### 3.8.1 Convex Functions

A function $f$ is **convex** if for all $\mathbf{x}, \mathbf{y}$ and $\lambda \in [0, 1]$:

$$f(\lambda\mathbf{x} + (1-\lambda)\mathbf{y}) \leq \lambda f(\mathbf{x}) + (1-\lambda)f(\mathbf{y})$$

Geometrically: the line segment between any two points on the graph lies above the graph. Equivalently, the Hessian $\mathbf{H} \succeq 0$ (positive semi-definite) everywhere.

**Key property:** A convex function has a single global minimum (or a convex set of minima). Gradient descent is guaranteed to find it.

**Convex loss functions in ML:** MSE (with linear models), logistic loss, hinge loss, L1/L2 regularization terms.

### 3.8.2 Non-Convex Optimization in Deep Learning

Neural network loss functions are highly non-convex due to:
1. **Nonlinear activations** (ReLU, sigmoid, etc.)
2. **Symmetries**: permuting neurons in a hidden layer gives the same function.
3. **Overparameterization**: many equivalent optima.

The loss landscape of a deep network contains:
- **Local minima**: points where all gradients are zero and the Hessian is positive definite.
- **Saddle points**: points where gradients are zero but the Hessian has both positive and negative eigenvalues.
- **Flat regions** (plateaus): areas where gradients are very small.

A surprising empirical finding (Dauphin et al., 2014; Choromanska et al., 2015) is that in high-dimensional spaces:
- Saddle points are far more common than local minima.
- Most local minima have loss values close to the global minimum.
- The "bad" local minima (with significantly higher loss) are exponentially rare.

This partially explains why gradient descent works well for deep networks despite non-convexity.

```python
# Visualizing a non-convex 2D loss landscape
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def loss_landscape(w1, w2):
    """A toy non-convex loss with multiple minima and saddle points."""
    return (np.sin(w1) * np.cos(w2) + 0.1 * (w1**2 + w2**2)
            + np.sin(3*w1) * np.sin(3*w2) * 0.5)

w1 = np.linspace(-4, 4, 200)
w2 = np.linspace(-4, 4, 200)
W1, W2 = np.meshgrid(w1, w2)
L = loss_landscape(W1, W2)

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(W1, W2, L, cmap='viridis', alpha=0.8)
ax1.set_xlabel('w1')
ax1.set_ylabel('w2')
ax1.set_zlabel('Loss')

ax2 = fig.add_subplot(122)
ax2.contourf(W1, W2, L, levels=50, cmap='viridis')
ax2.set_xlabel('w1')
ax2.set_ylabel('w2')
plt.colorbar(ax2.contourf(W1, W2, L, levels=50, cmap='viridis'))
plt.tight_layout()
```

---

## 3.9 Numerical Stability

Numerical stability is not a theoretical concern — it is a practical necessity. Floating-point arithmetic has finite precision, and naive implementations of common ML operations can produce NaN or Inf values, crashing training.

### 3.9.1 The Log-Sum-Exp Trick

Computing $\log \sum_i e^{x_i}$ (logsumexp) is ubiquitous in ML: it appears in softmax, cross-entropy, partition functions, and Bayesian model evidence. The naive computation overflows for large $x_i$ and underflows for very negative $x_i$.

**The trick:** Let $c = \max_i x_i$. Then:

$$\log \sum_i e^{x_i} = c + \log \sum_i e^{x_i - c}$$

Now the largest exponent is $e^0 = 1$, which cannot overflow.

```python
def logsumexp_naive(x):
    """Numerically unstable — DO NOT USE."""
    return np.log(np.sum(np.exp(x)))

def logsumexp_stable(x):
    """Numerically stable implementation."""
    c = np.max(x)
    return c + np.log(np.sum(np.exp(x - c)))

# Demonstration
x = np.array([1000, 1001, 1002])
# logsumexp_naive(x)   # inf (overflow in np.exp(1000))
print(logsumexp_stable(x))  # 1002.408... (correct)

# NumPy/SciPy provide this:
from scipy.special import logsumexp
print(logsumexp(x))  # 1002.408...
```

### 3.9.2 Stable Softmax

The softmax function $\sigma_i(\mathbf{z}) = \frac{e^{z_i}}{\sum_j e^{z_j}}$ suffers from the same overflow issue. The solution is to subtract the maximum before exponentiating:

$$\sigma_i(\mathbf{z}) = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_j e^{z_j - \max(\mathbf{z})}}$$

This is mathematically identical (the constant cancels) but numerically stable:

```python
def softmax_naive(z):
    """Unstable — overflows for large z."""
    exp_z = np.exp(z)
    return exp_z / exp_z.sum()

def softmax_stable(z):
    """Numerically stable softmax."""
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum()

z = np.array([1000.0, 1001.0, 1002.0])
# softmax_naive(z)    # [nan, nan, nan]
print(softmax_stable(z))  # [0.0900, 0.2447, 0.6652]
```

### 3.9.3 Gradient Clipping

When gradients become very large (exploding gradients), a single update can destroy the model. Gradient clipping limits the gradient magnitude:

**Clip by value:** Truncate each gradient element independently.
$$g_i = \text{clip}(g_i, -\tau, \tau)$$

**Clip by norm** (preferred): Scale the entire gradient vector if its norm exceeds a threshold, preserving the direction.

$$\mathbf{g} = \begin{cases} \mathbf{g} & \|\mathbf{g}\| \leq \tau \\ \frac{\tau}{\|\mathbf{g}\|} \mathbf{g} & \|\mathbf{g}\| > \tau \end{cases}$$

```python
def clip_grad_norm(gradients, max_norm):
    """Clip gradients by global norm."""
    total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        gradients = [g * clip_coef for g in gradients]
    return gradients, total_norm

# PyTorch built-in
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 3.10 Constrained Optimization

Many ML problems involve optimization with constraints: SVMs (margin maximization subject to classification constraints), variational autoencoders (KL divergence constraints), reinforcement learning (policy constraints), and entropy-based methods.

### 3.10.1 Lagrange Multipliers

For the problem:

$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{subject to} \quad g_i(\mathbf{x}) = 0, \quad i = 1, \ldots, m$$

The **Lagrangian** is:

$$\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}) = f(\mathbf{x}) + \sum_{i=1}^{m} \lambda_i g_i(\mathbf{x})$$

At the optimum, the following conditions hold:

$$\nabla_{\mathbf{x}} \mathcal{L} = \nabla f(\mathbf{x}) + \sum_{i=1}^{m} \lambda_i \nabla g_i(\mathbf{x}) = \mathbf{0}$$
$$g_i(\mathbf{x}) = 0, \quad \forall i$$

**Example: Maximum Entropy Distribution**

Find the probability distribution $\mathbf{p}$ that maximizes entropy subject to the constraint that probabilities sum to 1:

$$\max_{\mathbf{p}} -\sum_i p_i \ln p_i \quad \text{s.t.} \quad \sum_i p_i = 1$$

Lagrangian: $\mathcal{L} = -\sum_i p_i \ln p_i + \lambda\left(\sum_i p_i - 1\right)$

Taking derivatives: $\frac{\partial \mathcal{L}}{\partial p_i} = -\ln p_i - 1 + \lambda = 0$

This gives $p_i = e^{\lambda - 1}$, which is the same for all $i$. Using the constraint $\sum p_i = 1$: $p_i = 1/n$. The uniform distribution maximizes entropy — confirming the intuition that maximum uncertainty corresponds to maximum ignorance.

### 3.10.2 KKT Conditions

For problems with inequality constraints:

$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad g_i(\mathbf{x}) \leq 0, \quad h_j(\mathbf{x}) = 0$$

The **Karush-Kuhn-Tucker (KKT) conditions** are necessary conditions for optimality:

1. **Stationarity:** $\nabla f + \sum_i \mu_i \nabla g_i + \sum_j \lambda_j \nabla h_j = \mathbf{0}$
2. **Primal feasibility:** $g_i(\mathbf{x}) \leq 0$, $h_j(\mathbf{x}) = 0$
3. **Dual feasibility:** $\mu_i \geq 0$
4. **Complementary slackness:** $\mu_i g_i(\mathbf{x}) = 0$

Complementary slackness says: either the constraint is active ($g_i = 0$) or the multiplier is zero ($\mu_i = 0$). This is the mathematical foundation of support vector machines — only the "support vectors" (data points where the constraint is active) contribute to the solution.

**SVM derivation sketch:**

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1, \quad \forall i$$

The Lagrangian dual leads to:

$$\max_{\boldsymbol{\alpha}} \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j (\mathbf{x}_i \cdot \mathbf{x}_j) \quad \text{s.t.} \quad \alpha_i \geq 0, \quad \sum_i \alpha_i y_i = 0$$

By complementary slackness, $\alpha_i > 0$ only for support vectors (points on the margin).

---

## 3.11 Putting It All Together: A Complete Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Generate synthetic data
torch.manual_seed(42)
X = torch.randn(10000, 20)
true_w = torch.randn(20, 1)
y = X @ true_w + 0.1 * torch.randn(10000, 1)

# Train/val split
X_train, X_val = X[:8000], X[8000:]
y_train, y_val = y[:8000], y[8000:]

train_loader = DataLoader(
    TensorDataset(X_train, y_train), batch_size=64, shuffle=True
)

# Model
model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)

# AdamW optimizer with warmup+cosine schedule
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
total_steps = len(train_loader) * 50  # 50 epochs
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, total_steps=total_steps
)
criterion = nn.MSELoss()

# Training loop with gradient clipping
for epoch in range(50):
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item() * len(X_batch)

    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val).item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: train_loss={epoch_loss/len(X_train):.6f}, "
              f"val_loss={val_loss:.6f}, lr={scheduler.get_last_lr()[0]:.6f}")
```

---

## Exercises

### Conceptual

1. Derive the gradient of the binary cross-entropy loss with respect to the logits (pre-sigmoid values). Show that the result is simply $\sigma(z) - y$.

2. Explain why momentum helps in optimization by considering a 2D loss landscape shaped like a narrow valley. What would SGD trajectories look like vs. SGD with momentum?

3. Adam combines ideas from momentum and RMSProp. Explain why bias correction is necessary by computing the expected value of $m_1$ (after one step) when $m_0 = 0$.

4. Why does AdamW apply weight decay outside the adaptive learning rate mechanism? Give an example where L2 regularization within Adam would under-regularize certain parameters.

5. In a neural network with 100 layers, each layer's Jacobian has spectral norm 1.01. What happens to the gradient magnitude as it propagates backward? What if the spectral norm is 0.99?

### Programming

6. Implement the `Value` class from Section 3.3.3 and extend it with `sigmoid`, `tanh`, and `softmax` operations. Train a simple logistic regression model on a 2D classification dataset.

7. Implement SGD, Momentum, RMSProp, Adam, and AdamW from scratch (as shown in this chapter). Compare them on the Rosenbrock function $f(x,y) = (1-x)^2 + 100(y-x^2)^2$. Plot the optimization trajectories and convergence curves.

8. Implement the warmup + cosine decay learning rate schedule. Train a 3-layer MLP on MNIST with and without the schedule. Plot the learning rate over time and the validation accuracy curves.

9. Numerically verify that the Hessian of the logistic regression loss is positive semi-definite for any dataset. Generate random data, compute the Hessian using `torch.autograd.functional.hessian`, and check that all eigenvalues are non-negative.

10. Implement gradient clipping by norm and gradient clipping by value. On a synthetic regression task with outliers (which cause exploding gradients), show that gradient clipping stabilizes training while no clipping causes divergence.

---

## References

- Choromanska, A., Henaff, M., Mathieu, M., Arous, G. B., & LeCun, Y. (2015). The Loss Surfaces of Multilayer Networks. *AISTATS*.
- Dauphin, Y. N., Pascanu, R., Gulcehre, C., Cho, K., Ganguli, S., & Bengio, Y. (2014). Identifying and attacking the saddle point problem in high-dimensional non-convex optimization. *NeurIPS*.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapters 4-8. MIT Press.
- Hinton, G. (2012). Neural Networks for Machine Learning, Lecture 6a: Overview of mini-batch gradient descent. Coursera.
- Karpathy, A. (2022). micrograd: A tiny autograd engine. https://github.com/karpathy/micrograd
- Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR*.
- Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. *ICLR*.
- Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR*.
- Nesterov, Y. (1983). A method for solving the convex programming problem with convergence rate O(1/k^2). *Soviet Mathematics Doklady*.
- Ruder, S. (2016). An overview of gradient descent optimization algorithms. *arXiv preprint arXiv:1609.04747*.
- Smith, L. N., & Topin, N. (2019). Super-convergence: very fast training of neural networks using large learning rates. *Artificial Intelligence and Machine Learning for Multi-Domain Operations Applications*.
- Stanford CS231n. (2023). Convolutional Neural Networks for Visual Recognition: Optimization notes. https://cs231n.github.io/

---

*Next chapter: Probability and Statistics — the mathematical framework for reasoning about uncertainty, the foundation of every learning algorithm.*
