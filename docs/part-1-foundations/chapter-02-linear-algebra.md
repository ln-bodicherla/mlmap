# Chapter 2: Linear Algebra for Machine Learning

> *"The miracle of the matrix is not that it can represent a linear transformation — it is that it can represent any linear transformation."*
> — Gilbert Strang

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Perform vector and matrix operations with geometric intuition, understanding dot products, projections, and transformations.
2. Solve systems of linear equations using Gaussian elimination and understand when solutions exist.
3. Reason about vector spaces, subspaces, basis, dimension, and rank.
4. Compute and interpret eigenvalues and eigenvectors, connecting them to matrix behavior.
5. Decompose matrices using Singular Value Decomposition (SVD) and apply it to dimensionality reduction, recommendation systems, and low-rank approximation.
6. Distinguish between matrix norms (L1, L2, Frobenius, spectral) and explain their role in regularization.
7. Work with higher-order tensors and use Einstein summation notation efficiently.
8. Implement all concepts in NumPy and PyTorch with production-quality code.
9. Derive PCA from first principles using the eigendecomposition framework.
10. Connect linear algebra to real ML applications including word embeddings, image compression, and LoRA fine-tuning.

---

## 2.0 Why Linear Algebra Matters for ML

Linear algebra is not merely a prerequisite for machine learning — it *is* the language of machine learning. Every major ML operation can be expressed as a linear algebra operation:

- A **linear regression** prediction is a dot product: $\hat{y} = \mathbf{w}^T\mathbf{x} + b$.
- A **neural network layer** is a matrix multiplication followed by a nonlinearity: $\mathbf{h} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$.
- **Attention** in transformers computes scaled dot products between query and key matrices: $\text{softmax}(\mathbf{QK}^T / \sqrt{d})$.
- **PCA** is an eigendecomposition of the covariance matrix.
- **LoRA fine-tuning** adds a low-rank matrix to pretrained weights.
- **Batch normalization** subtracts the mean and divides by the standard deviation — operations on vectors.

A practitioner who does not understand linear algebra can use ML frameworks mechanically but cannot debug shape mismatches, understand why a model is slow, choose appropriate regularization, or read a research paper. Linear algebra is the difference between using ML and understanding it.

This chapter builds from vectors through matrices to the powerful decomposition theorems (eigendecomposition, SVD) that underpin dimensionality reduction, recommendation systems, and modern parameter-efficient fine-tuning. We emphasize both mathematical rigor and computational practice, implementing every concept in NumPy and PyTorch (Strang, 2016).

---

## 2.1 Vectors: The Building Blocks

At its core, machine learning is about learning functions from data. That data — images, text, audio, tabular records — is represented as vectors. A single data point is a vector; a dataset is a collection of vectors. Understanding vectors geometrically and algebraically is therefore the first step toward understanding machine learning mathematically.

### 2.1.1 Vectors as Geometry and as Data

A **vector** $\mathbf{v} \in \mathbb{R}^n$ is an ordered list of $n$ real numbers:

$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$$

Geometrically, a vector in $\mathbb{R}^2$ or $\mathbb{R}^3$ is an arrow from the origin to a point in space. In machine learning, we work with vectors in $\mathbb{R}^{768}$ (BERT embeddings), $\mathbb{R}^{4096}$ (LLM hidden states), or even $\mathbb{R}^{196608}$ (a 256x256 RGB image flattened). The geometric intuition from 2D and 3D generalizes to these high-dimensional spaces.

### 2.1.2 Vector Operations

**Addition and scalar multiplication** are defined element-wise:

$$\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{bmatrix}, \quad c\mathbf{v} = \begin{bmatrix} cv_1 \\ cv_2 \\ \vdots \\ cv_n \end{bmatrix}$$

These two operations define a **vector space**: any expression of the form $c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k$ (a **linear combination**) is also a vector in the space.

**The dot product** (inner product) is the single most important operation in machine learning:

$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta$$

where $\theta$ is the angle between the vectors and $\|\mathbf{v}\| = \sqrt{\sum_i v_i^2}$ is the Euclidean norm (length).

The dot product appears everywhere in ML:
- **Linear regression**: prediction is $\hat{y} = \mathbf{w} \cdot \mathbf{x} + b$
- **Attention mechanism**: similarity score is $\mathbf{q} \cdot \mathbf{k}$
- **Cosine similarity**: $\cos\theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$

```python
import numpy as np

u = np.array([1.0, 2.0, 3.0])
v = np.array([4.0, 5.0, 6.0])

# Dot product — three equivalent ways
dot1 = np.dot(u, v)         # 32.0
dot2 = u @ v                # 32.0 (preferred syntax)
dot3 = np.sum(u * v)        # 32.0

# Norm (length)
norm_u = np.linalg.norm(u)  # sqrt(14) ≈ 3.742

# Cosine similarity
cosine_sim = (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))
# 0.9746
```

### 2.1.3 Projections

The **projection** of $\mathbf{u}$ onto $\mathbf{v}$ gives the component of $\mathbf{u}$ in the direction of $\mathbf{v}$:

$$\text{proj}_{\mathbf{v}} \mathbf{u} = \frac{\mathbf{u} \cdot \mathbf{v}}{\mathbf{v} \cdot \mathbf{v}} \mathbf{v}$$

Projections are fundamental to:
- **Least squares regression**: projecting the target vector onto the column space of the feature matrix
- **PCA**: projecting data onto principal component directions
- **Gram-Schmidt orthogonalization**: building orthonormal bases

```python
def project(u, v):
    """Project vector u onto vector v."""
    return (u @ v) / (v @ v) * v

u = np.array([3.0, 4.0])
v = np.array([1.0, 0.0])  # x-axis
proj = project(u, v)       # [3.0, 0.0] — the x-component of u
```

---

## 2.2 Matrices: Linear Transformations in Disguise

A **matrix** $\mathbf{A} \in \mathbb{R}^{m \times n}$ is a rectangular array of numbers with $m$ rows and $n$ columns:

$$\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}$$

But a matrix is much more than a table of numbers. It is a **linear transformation** — a function that takes vectors in $\mathbb{R}^n$ as input and produces vectors in $\mathbb{R}^m$ as output, while preserving the structure of addition and scalar multiplication.

### 2.2.1 Matrix Operations

**Matrix-vector multiplication** $\mathbf{y} = \mathbf{A}\mathbf{x}$ where $\mathbf{A} \in \mathbb{R}^{m \times n}$, $\mathbf{x} \in \mathbb{R}^n$, produces $\mathbf{y} \in \mathbb{R}^m$:

$$y_i = \sum_{j=1}^{n} A_{ij} x_j$$

This can be understood two ways:
1. **Row perspective**: each $y_i$ is the dot product of row $i$ of $\mathbf{A}$ with $\mathbf{x}$.
2. **Column perspective**: $\mathbf{y}$ is a linear combination of the columns of $\mathbf{A}$, with coefficients given by $\mathbf{x}$.

**Matrix-matrix multiplication** $\mathbf{C} = \mathbf{A}\mathbf{B}$ where $\mathbf{A} \in \mathbb{R}^{m \times p}$, $\mathbf{B} \in \mathbb{R}^{p \times n}$ produces $\mathbf{C} \in \mathbb{R}^{m \times n}$:

$$C_{ij} = \sum_{k=1}^{p} A_{ik} B_{kj}$$

Matrix multiplication represents the **composition** of linear transformations: if $\mathbf{A}$ represents transformation $f$ and $\mathbf{B}$ represents transformation $g$, then $\mathbf{AB}$ represents $f \circ g$ (apply $g$ first, then $f$).

```python
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
x = np.random.randn(4)

# Matrix-vector multiplication
y = A @ x           # shape: (3,)

# Matrix-matrix multiplication
C = A @ B           # shape: (3, 5)

# Transpose
A_T = A.T           # shape: (4, 3)

# Key identity: (AB)^T = B^T A^T
assert np.allclose((A @ B).T, B.T @ A.T)
```

### 2.2.2 Special Matrix Types

Several types of matrices appear repeatedly in ML:

**Symmetric matrix**: $\mathbf{A} = \mathbf{A}^T$. Covariance matrices, kernel matrices, and Hessians are symmetric. Symmetric matrices have real eigenvalues and orthogonal eigenvectors (the Spectral Theorem).

**Orthogonal matrix**: $\mathbf{Q}^T\mathbf{Q} = \mathbf{Q}\mathbf{Q}^T = \mathbf{I}$. Orthogonal matrices preserve lengths and angles. Rotations and reflections are orthogonal transformations. They appear in SVD, QR decomposition, and as initialization strategies for neural networks (Saxe et al., 2014).

**Diagonal matrix**: $D_{ij} = 0$ for $i \neq j$. Multiplication by a diagonal matrix simply scales each dimension independently. Diagonal matrices appear in SVD (the singular values) and in the eigendecomposition.

**Positive definite matrix**: $\mathbf{x}^T\mathbf{A}\mathbf{x} > 0$ for all $\mathbf{x} \neq \mathbf{0}$. Covariance matrices are positive semi-definite. The Hessian of a convex function is positive semi-definite. Positive definiteness guarantees that gradient descent on a quadratic converges to a unique minimum.

```python
# Symmetric matrix (covariance)
X = np.random.randn(100, 5)
cov = X.T @ X / len(X)    # Always symmetric
assert np.allclose(cov, cov.T)

# Orthogonal matrix (rotation in 2D)
theta = np.pi / 4  # 45 degrees
Q = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
assert np.allclose(Q.T @ Q, np.eye(2))

# Check positive definiteness
eigenvalues = np.linalg.eigvalsh(cov)
is_pos_def = np.all(eigenvalues > 0)
```

### 2.2.3 The Inverse and Pseudo-Inverse

The **inverse** $\mathbf{A}^{-1}$ of a square matrix $\mathbf{A}$ satisfies $\mathbf{A}^{-1}\mathbf{A} = \mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$. It exists if and only if $\mathbf{A}$ is non-singular (full rank, nonzero determinant).

The **Moore-Penrose pseudo-inverse** $\mathbf{A}^+$ generalizes the inverse to non-square and singular matrices. For a full-rank matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ with $m > n$ (overdetermined system), the pseudo-inverse gives the least-squares solution:

$$\mathbf{A}^+ = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T$$

This is exactly the normal equation for linear regression:

$$\hat{\mathbf{w}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

```python
# Solving linear regression with the normal equation
X = np.c_[np.ones(100), np.random.randn(100, 3)]  # Add bias column
y = X @ np.array([2.0, 0.5, -1.0, 3.0]) + 0.1 * np.random.randn(100)

# Method 1: Normal equation
w = np.linalg.inv(X.T @ X) @ X.T @ y

# Method 2: Pseudo-inverse (more numerically stable)
w = np.linalg.pinv(X) @ y

# Method 3: Least squares solver (most numerically stable)
w, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
```

### 2.2.4 The Determinant

The **determinant** of a square matrix $\mathbf{A}$ is a scalar value that encodes important geometric and algebraic information.

For a $2 \times 2$ matrix: $\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$

Geometric interpretation: $|\det(\mathbf{A})|$ is the factor by which $\mathbf{A}$ scales areas (in 2D) or volumes (in 3D). If $\det(\mathbf{A}) = 0$, the transformation collapses a dimension (the matrix is singular).

Key properties:
- $\det(\mathbf{AB}) = \det(\mathbf{A})\det(\mathbf{B})$
- $\det(\mathbf{A}^T) = \det(\mathbf{A})$
- $\det(\mathbf{A}^{-1}) = 1/\det(\mathbf{A})$
- $\det(c\mathbf{A}) = c^n\det(\mathbf{A})$ for an $n \times n$ matrix
- $\det(\mathbf{A}) = \prod_i \lambda_i$ (product of eigenvalues)

**ML connection:** The log-determinant of the covariance matrix appears in the multivariate Gaussian likelihood:

$$\log p(\mathbf{x}) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\log|\boldsymbol{\Sigma}| - \frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})$$

Computing $\log|\boldsymbol{\Sigma}|$ is critical for Gaussian mixture models, variational inference, and normalizing flows.

```python
A = np.array([[3, 1], [2, 4]])
print(np.linalg.det(A))  # 10.0 (= 3*4 - 1*2)

# For positive definite matrices, use Cholesky for stable log-determinant
L = np.linalg.cholesky(cov_matrix)
log_det = 2 * np.sum(np.log(np.diag(L)))  # More stable than np.log(np.linalg.det(...))
```

### 2.2.5 Matrix Decompositions Overview

Matrix decompositions are the Swiss army knife of numerical linear algebra. Each decomposes a matrix into simpler factors for different purposes:

| Decomposition | Form | Use in ML |
|---|---|---|
| LU | $\mathbf{A} = \mathbf{LU}$ | Solving linear systems |
| Cholesky | $\mathbf{A} = \mathbf{LL}^T$ | Gaussian processes, sampling |
| QR | $\mathbf{A} = \mathbf{QR}$ | Least squares, Gram-Schmidt |
| Eigendecomposition | $\mathbf{A} = \mathbf{V\Lambda V}^{-1}$ | PCA, spectral methods |
| SVD | $\mathbf{A} = \mathbf{U\Sigma V}^T$ | Everything (universal workhorse) |

The **Cholesky decomposition** $\mathbf{A} = \mathbf{LL}^T$ (where $\mathbf{L}$ is lower triangular) exists for positive definite matrices. It is twice as fast as LU and is the standard way to:
- Sample from a multivariate Gaussian: $\mathbf{x} = \boldsymbol{\mu} + \mathbf{L}\mathbf{z}$ where $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
- Solve $\boldsymbol{\Sigma}\mathbf{x} = \mathbf{b}$ via forward and back substitution
- Compute log-determinants stably

```python
# Cholesky decomposition
cov = np.array([[4, 2], [2, 3]])
L = np.linalg.cholesky(cov)
print(L)
# [[2.   0.  ]
#  [1.   1.41]]
assert np.allclose(L @ L.T, cov)

# Sample from multivariate Gaussian using Cholesky
mean = np.array([1.0, 2.0])
z = np.random.randn(1000, 2)
samples = mean + z @ L.T  # Each row is a sample from N(mean, cov)
```

---

## 2.3 Systems of Linear Equations

A system of $m$ linear equations in $n$ unknowns can be written as $\mathbf{A}\mathbf{x} = \mathbf{b}$:

$$\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} = \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_m \end{bmatrix}$$

### 2.3.1 Gaussian Elimination

Gaussian elimination systematically transforms $[\mathbf{A} | \mathbf{b}]$ (the augmented matrix) into row echelon form using three elementary row operations:
1. Swap two rows
2. Multiply a row by a nonzero scalar
3. Add a multiple of one row to another

```python
def gaussian_elimination(A, b):
    """Solve Ax = b using Gaussian elimination with partial pivoting."""
    n = len(b)
    # Augmented matrix
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    # Forward elimination
    for col in range(n):
        # Partial pivoting: find the largest element in the column
        max_row = np.argmax(np.abs(Ab[col:, col])) + col
        Ab[[col, max_row]] = Ab[[max_row, col]]  # Swap rows

        if np.abs(Ab[col, col]) < 1e-12:
            raise ValueError("Matrix is singular or nearly singular")

        # Eliminate below
        for row in range(col + 1, n):
            factor = Ab[row, col] / Ab[col, col]
            Ab[row, col:] -= factor * Ab[col, col:]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - Ab[i, i+1:n] @ x[i+1:n]) / Ab[i, i]

    return x

# Test
A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
b = np.array([8, -11, -3], dtype=float)
x = gaussian_elimination(A, b)
print(x)  # [2, 3, -1]
assert np.allclose(A @ x, b)
```

**Three possible outcomes** for $\mathbf{A}\mathbf{x} = \mathbf{b}$:
1. **Unique solution**: $\mathbf{A}$ is square and full rank.
2. **No solution**: the system is inconsistent ($\mathbf{b}$ is not in the column space of $\mathbf{A}$).
3. **Infinitely many solutions**: $\mathbf{A}$ has more columns than its rank (underdetermined).

In ML, we almost always deal with case 2 (overdetermined systems — more data than parameters), which is why we minimize $\|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2$ (least squares) rather than solving exactly.

---

## 2.4 Vector Spaces, Subspaces, Basis, and Rank

### 2.4.1 Vector Spaces and Subspaces

A **vector space** $V$ is a set of vectors closed under addition and scalar multiplication. $\mathbb{R}^n$ is the prototypical example.

A **subspace** $S \subseteq V$ is a subset that is itself a vector space (must contain the zero vector, must be closed under addition and scalar multiplication).

The most important subspaces associated with a matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ are:
- **Column space** $\text{Col}(\mathbf{A})$: all vectors $\mathbf{b}$ for which $\mathbf{A}\mathbf{x} = \mathbf{b}$ has a solution.
- **Null space** $\text{Null}(\mathbf{A})$: all vectors $\mathbf{x}$ satisfying $\mathbf{A}\mathbf{x} = \mathbf{0}$.
- **Row space** $\text{Row}(\mathbf{A})$: the column space of $\mathbf{A}^T$.

### 2.4.2 Basis and Dimension

A **basis** for a vector space $V$ is a set of linearly independent vectors that span $V$. Every vector in $V$ can be written as a unique linear combination of the basis vectors.

The **dimension** of a space is the number of vectors in any basis.

The **rank** of a matrix is the dimension of its column space (equivalently, its row space). It equals the number of linearly independent columns (or rows).

$$\text{rank}(\mathbf{A}) + \text{nullity}(\mathbf{A}) = n \quad \text{(Rank-Nullity Theorem)}$$

where nullity is the dimension of the null space and $n$ is the number of columns.

```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

rank = np.linalg.matrix_rank(A)  # 2 (rows are linearly dependent)

# The null space: vectors x such that Ax = 0
# For this matrix, [1, -2, 1] is in the null space:
x_null = np.array([1, -2, 1])
print(A @ x_null)  # [0, 0, 0] (up to numerical precision)
```

**ML connection**: The rank of the data matrix tells us the intrinsic dimensionality of the data. If a 1000-dimensional dataset has rank 50, it actually lives in a 50-dimensional subspace. This is the insight behind PCA, autoencoders, and low-rank factorization.

### 2.4.3 The Four Fundamental Subspaces

Gilbert Strang's "four fundamental subspaces" (Strang, 2016) provide a complete picture of what a matrix does. For $\mathbf{A} \in \mathbb{R}^{m \times n}$ with rank $r$:

| Subspace | Dimension | Space |
|---|---|---|
| Column space $\text{Col}(\mathbf{A})$ | $r$ | $\mathbb{R}^m$ |
| Row space $\text{Row}(\mathbf{A})$ | $r$ | $\mathbb{R}^n$ |
| Null space $\text{Null}(\mathbf{A})$ | $n - r$ | $\mathbb{R}^n$ |
| Left null space $\text{Null}(\mathbf{A}^T)$ | $m - r$ | $\mathbb{R}^m$ |

The key orthogonality relationships:
- Row space $\perp$ Null space (in $\mathbb{R}^n$)
- Column space $\perp$ Left null space (in $\mathbb{R}^m$)

**ML interpretation:** In linear regression $\mathbf{X}\mathbf{w} = \mathbf{y}$:
- The column space of $\mathbf{X}$ is the set of all possible predictions.
- The residual $\mathbf{y} - \hat{\mathbf{y}}$ lies in the left null space (perpendicular to all predictions).
- If the null space is nontrivial ($n > r$), there are infinitely many optimal weight vectors — the model is underdetermined. This is the normal state of affairs for overparameterized deep networks.

```python
from scipy.linalg import null_space

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Column space basis (via SVD)
U, sigma, Vt = np.linalg.svd(A)
rank = np.sum(sigma > 1e-10)
col_space_basis = U[:, :rank]

# Null space
ns = null_space(A)
print(f"Null space dimension: {ns.shape[1]}")  # 1
print(f"Null space basis:\n{ns}")
# Verify: A @ ns ≈ 0
print(f"A @ null_vector = {A @ ns[:, 0]}")  # ≈ [0, 0, 0]
```

### 2.4.4 Linear Independence and the Gram Matrix

A set of vectors $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ is **linearly independent** if the only solution to $c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k = \mathbf{0}$ is $c_1 = \cdots = c_k = 0$.

The **Gram matrix** $\mathbf{G}$ with entries $G_{ij} = \mathbf{v}_i^T\mathbf{v}_j$ encodes all pairwise inner products. The vectors are linearly independent if and only if $\det(\mathbf{G}) \neq 0$.

Gram matrices appear in:
- **Kernel methods**: the kernel matrix $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$ is a generalized Gram matrix.
- **Style transfer**: the Gram matrix of CNN feature maps captures texture and style (Gatys et al., 2016).
- **Gaussian processes**: the covariance matrix is a kernel/Gram matrix.

---

## 2.5 Linear Transformations

A function $T: \mathbb{R}^n \to \mathbb{R}^m$ is a **linear transformation** if:
1. $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$ (additivity)
2. $T(c\mathbf{u}) = cT(\mathbf{u})$ (homogeneity)

Every linear transformation from $\mathbb{R}^n$ to $\mathbb{R}^m$ can be represented by an $m \times n$ matrix. Conversely, every matrix defines a linear transformation.

Common geometric transformations as matrices in 2D:

$$\text{Rotation by } \theta: \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}, \quad \text{Scaling: } \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}, \quad \text{Reflection over x-axis: } \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$$

**ML connection**: A fully connected neural network layer performs an **affine transformation**: $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$. The weight matrix $\mathbf{W}$ is a linear transformation; the bias $\mathbf{b}$ makes it affine. Nonlinear activation functions (ReLU, GELU) are what give neural networks their expressive power beyond linear transformations.

```python
import torch

# A neural network layer IS a linear transformation + nonlinearity
layer = torch.nn.Linear(in_features=768, out_features=256)

# The weight matrix defines the linear transformation
print(layer.weight.shape)  # (256, 768) — maps R^768 -> R^256

# Forward pass: y = Wx + b, then activation
x = torch.randn(32, 768)  # batch of 32 vectors
y = torch.relu(layer(x))  # (32, 256)
```

---

## 2.6 Eigenvalues and Eigenvectors

Eigenvalues and eigenvectors reveal the intrinsic structure of a linear transformation. They tell us which directions are preserved (only scaled) under the transformation.

### 2.6.1 Definition and Geometric Intuition

A nonzero vector $\mathbf{v}$ is an **eigenvector** of $\mathbf{A}$ with **eigenvalue** $\lambda$ if:

$$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$

This means that $\mathbf{A}$ acts on $\mathbf{v}$ by simply scaling it — the direction is preserved. The eigenvalue $\lambda$ is the scaling factor.

Geometrically (Sanderson, 2016):
- If $\lambda > 1$: the eigenvector is stretched.
- If $0 < \lambda < 1$: it is compressed.
- If $\lambda < 0$: it is flipped and scaled.
- If $\lambda = 0$: the eigenvector is mapped to zero (the matrix is singular).

### 2.6.2 Computing Eigenvalues

Eigenvalues are the roots of the **characteristic polynomial**:

$$\det(\mathbf{A} - \lambda\mathbf{I}) = 0$$

For a $2 \times 2$ matrix $\mathbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$:

$$\lambda^2 - (a+d)\lambda + (ad - bc) = 0$$
$$\lambda = \frac{(a+d) \pm \sqrt{(a+d)^2 - 4(ad-bc)}}{2}$$

The trace $\text{tr}(\mathbf{A}) = a + d = \sum_i \lambda_i$ is the sum of eigenvalues. The determinant $\det(\mathbf{A}) = ad - bc = \prod_i \lambda_i$ is the product of eigenvalues.

### 2.6.3 The Eigendecomposition

A diagonalizable matrix $\mathbf{A}$ can be factored as:

$$\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}$$

where $\mathbf{V}$ is the matrix of eigenvectors (as columns) and $\boldsymbol{\Lambda}$ is the diagonal matrix of eigenvalues.

For symmetric matrices (which include covariance matrices, kernel matrices, and Hessians), the **Spectral Theorem** guarantees:
1. All eigenvalues are real.
2. Eigenvectors corresponding to distinct eigenvalues are orthogonal.
3. $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$ where $\mathbf{Q}$ is orthogonal.

```python
# Eigendecomposition
A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")    # [5., 2.]
print(f"Eigenvectors:\n{eigenvectors}") # columns are eigenvectors

# Verify: Av = λv
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    assert np.allclose(A @ v, lam * v)

# Symmetric matrix — use eigh (faster, more stable)
S = np.array([[3, 1], [1, 2]])
eigenvalues, eigenvectors = np.linalg.eigh(S)  # Guaranteed real, orthogonal
```

### 2.6.4 ML Applications of Eigendecomposition

**PCA (Principal Component Analysis)**: The eigenvectors of the data covariance matrix are the principal components — the directions of maximum variance. The corresponding eigenvalues tell us how much variance each component captures.

**Graph Laplacian and spectral clustering**: The eigenvectors of the graph Laplacian matrix reveal cluster structure.

**Stability of dynamical systems**: In recurrent neural networks, eigenvalues of the weight matrix determine whether signals grow (exploding gradients, $|\lambda| > 1$) or shrink (vanishing gradients, $|\lambda| < 1$).

---

## 2.7 Singular Value Decomposition (SVD)

The Singular Value Decomposition is arguably the most important matrix factorization in applied mathematics. It generalizes the eigendecomposition to any matrix (not just square ones) and provides the optimal low-rank approximation.

### 2.7.1 The Decomposition

Any matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ can be decomposed as:

$$\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$$

where:
- $\mathbf{U} \in \mathbb{R}^{m \times m}$ is orthogonal (columns are **left singular vectors**)
- $\boldsymbol{\Sigma} \in \mathbb{R}^{m \times n}$ is diagonal with non-negative entries $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$ (the **singular values**)
- $\mathbf{V} \in \mathbb{R}^{n \times n}$ is orthogonal (columns are **right singular vectors**)

The singular values are the square roots of the eigenvalues of $\mathbf{A}^T\mathbf{A}$ (or equivalently $\mathbf{A}\mathbf{A}^T$).

### 2.7.2 Geometric Interpretation

The SVD tells us that every linear transformation can be decomposed into three steps:
1. **Rotate** (by $\mathbf{V}^T$): align with the "natural" coordinate system of the transformation.
2. **Scale** (by $\boldsymbol{\Sigma}$): stretch or compress along each axis.
3. **Rotate** (by $\mathbf{U}$): map to the output coordinate system.

This is profound: *any* linear transformation — shearing, projection, whatever — is just rotation-scaling-rotation.

```python
A = np.array([[3, 2, 2],
              [2, 3, -2]])

U, sigma, Vt = np.linalg.svd(A, full_matrices=True)
print(f"U shape: {U.shape}")       # (2, 2)
print(f"Sigma: {sigma}")           # [5., 3.]
print(f"V^T shape: {Vt.shape}")    # (3, 3)

# Reconstruct A from SVD
Sigma_full = np.zeros_like(A, dtype=float)
np.fill_diagonal(Sigma_full, sigma)
A_reconstructed = U @ Sigma_full @ Vt
assert np.allclose(A, A_reconstructed)
```

### 2.7.3 Truncated SVD and Low-Rank Approximation

The **Eckart-Young-Mirsky theorem** states that the best rank-$k$ approximation to $\mathbf{A}$ (in both the Frobenius and spectral norms) is obtained by keeping only the top $k$ singular values:

$$\mathbf{A}_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T = \mathbf{U}_k \boldsymbol{\Sigma}_k \mathbf{V}_k^T$$

The approximation error is:

$$\|\mathbf{A} - \mathbf{A}_k\|_F = \sqrt{\sigma_{k+1}^2 + \sigma_{k+2}^2 + \cdots + \sigma_r^2}$$

This result is the mathematical foundation for:
- **Dimensionality reduction** (PCA is SVD applied to centered data)
- **Image compression** (keep top $k$ singular values of the pixel matrix)
- **Latent Semantic Analysis** (truncated SVD on term-document matrices)
- **LoRA** (Low-Rank Adaptation of Large Language Models) (Hu et al., 2021)

```python
# Image compression with truncated SVD
from PIL import Image
import matplotlib.pyplot as plt

# Load grayscale image as matrix
img = np.array(Image.open("photo.jpg").convert("L"), dtype=float)
print(f"Original shape: {img.shape}")  # e.g., (480, 640)

U, sigma, Vt = np.linalg.svd(img, full_matrices=False)

# Reconstruct with different ranks
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, k in zip(axes, [5, 20, 50, 200]):
    img_k = (U[:, :k] * sigma[:k]) @ Vt[:k, :]
    ax.imshow(img_k, cmap="gray")
    ax.set_title(f"Rank {k}")
    # Compression ratio
    original_size = img.shape[0] * img.shape[1]
    compressed_size = k * (img.shape[0] + img.shape[1] + 1)
    ax.set_xlabel(f"Compression: {compressed_size/original_size:.1%}")
plt.tight_layout()
plt.show()
```

### 2.7.4 SVD in Recommendation Systems

Collaborative filtering via matrix factorization is one of the most successful applications of SVD in industry. The user-item interaction matrix $\mathbf{R} \in \mathbb{R}^{m \times n}$ (with many missing entries) is approximated as:

$$\mathbf{R} \approx \mathbf{U}_k \boldsymbol{\Sigma}_k \mathbf{V}_k^T$$

The rows of $\mathbf{U}_k$ are user embeddings and the columns of $\mathbf{V}_k^T$ are item embeddings, both in a shared $k$-dimensional latent space.

```python
from scipy.sparse.linalg import svds

# Sparse user-item matrix (e.g., movie ratings)
# R[i, j] = rating that user i gave to item j (0 if unknown)
R = load_ratings_matrix()  # shape: (n_users, n_items)

# Mean-center (per user)
user_means = np.array(R.mean(axis=1)).flatten()
R_centered = R - user_means[:, np.newaxis]

# Truncated SVD with k latent factors
k = 50
U, sigma, Vt = svds(R_centered, k=k)

# Predict missing ratings
predicted_ratings = U @ np.diag(sigma) @ Vt + user_means[:, np.newaxis]
```

### 2.7.5 SVD and LoRA

LoRA (Low-Rank Adaptation) is a technique for fine-tuning large language models by adding low-rank updates to the weight matrices (Hu et al., 2021). Instead of updating a full weight matrix $\mathbf{W} \in \mathbb{R}^{d \times d}$ (which might have millions of parameters), LoRA parameterizes the update as:

$$\mathbf{W}' = \mathbf{W} + \mathbf{B}\mathbf{A}$$

where $\mathbf{B} \in \mathbb{R}^{d \times r}$ and $\mathbf{A} \in \mathbb{R}^{r \times d}$ with $r \ll d$. The product $\mathbf{B}\mathbf{A}$ is a rank-$r$ matrix, capturing the essential adaptation with far fewer trainable parameters.

The mathematical insight is that the weight update during fine-tuning has low intrinsic rank — it can be well-approximated by a low-rank matrix, which is exactly what truncated SVD tells us is optimal.

---

## 2.8 Matrix Norms

Matrix norms measure the "size" of a matrix and play a crucial role in regularization, convergence analysis, and numerical stability.

### 2.8.1 Vector Norms

The $L_p$ norm of a vector $\mathbf{x} \in \mathbb{R}^n$ is:

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^{n} |x_i|^p\right)^{1/p}$$

Key cases:
- **L1 norm** (Manhattan): $\|\mathbf{x}\|_1 = \sum |x_i|$ — promotes sparsity in weights (Lasso regularization).
- **L2 norm** (Euclidean): $\|\mathbf{x}\|_2 = \sqrt{\sum x_i^2}$ — penalizes large weights smoothly (Ridge regularization).
- **L-infinity norm**: $\|\mathbf{x}\|_\infty = \max|x_i|$ — the maximum absolute value.

### 2.8.2 Matrix Norms

**Frobenius norm**: the "Euclidean norm for matrices":

$$\|\mathbf{A}\|_F = \sqrt{\sum_{i,j} A_{ij}^2} = \sqrt{\text{tr}(\mathbf{A}^T\mathbf{A})} = \sqrt{\sum_i \sigma_i^2}$$

**Spectral norm** (operator 2-norm): the largest singular value:

$$\|\mathbf{A}\|_2 = \sigma_{\max}(\mathbf{A}) = \max_{\|\mathbf{x}\|=1} \|\mathbf{A}\mathbf{x}\|$$

This measures the maximum "stretching" the transformation can do. It is used in spectral normalization for GANs (Miyato et al., 2018).

**Nuclear norm**: the sum of singular values:

$$\|\mathbf{A}\|_* = \sum_i \sigma_i$$

The nuclear norm is the convex relaxation of the matrix rank, used in matrix completion problems.

```python
A = np.random.randn(4, 3)

# Frobenius norm
frob = np.linalg.norm(A, 'fro')
frob_manual = np.sqrt(np.sum(A**2))
assert np.isclose(frob, frob_manual)

# Spectral norm (largest singular value)
spectral = np.linalg.norm(A, 2)
_, sigma, _ = np.linalg.svd(A)
assert np.isclose(spectral, sigma[0])

# Nuclear norm (sum of singular values)
nuclear = np.linalg.norm(A, 'nuc')
assert np.isclose(nuclear, np.sum(sigma))
```

### 2.8.3 Norms in Regularization

Regularization adds a norm penalty to the loss function to prevent overfitting:

$$\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{data}} + \lambda \|\mathbf{w}\|_p^p$$

- **L2 regularization (Ridge / weight decay)**: $\lambda \|\mathbf{w}\|_2^2 = \lambda \sum w_i^2$. Shrinks weights toward zero. Equivalent to a Gaussian prior on weights (MAP estimation).
- **L1 regularization (Lasso)**: $\lambda \|\mathbf{w}\|_1 = \lambda \sum |w_i|$. Drives some weights exactly to zero, performing feature selection. Equivalent to a Laplace prior on weights.
- **Elastic Net**: $\lambda_1 \|\mathbf{w}\|_1 + \lambda_2 \|\mathbf{w}\|_2^2$. Combines the benefits of both.

---

## 2.9 Tensors and Einstein Summation

### 2.9.1 Higher-Order Tensors

A **tensor** is a generalization of scalars (order 0), vectors (order 1), and matrices (order 2) to higher orders. In deep learning:
- A grayscale image is a 2D tensor (height x width)
- An RGB image is a 3D tensor (channels x height x width)
- A batch of RGB images is a 4D tensor (batch x channels x height x width)
- A video is a 5D tensor (batch x time x channels x height x width)
- Attention weights in a multi-head transformer are 4D (batch x heads x seq_len x seq_len)

### 2.9.2 Einstein Summation (einsum)

Einstein summation notation provides a compact, general way to express tensor contractions, outer products, traces, and more. The convention: repeated indices are summed over.

```python
import numpy as np
import torch

# Matrix multiplication: C_ij = sum_k A_ik B_kj
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
C = np.einsum('ik,kj->ij', A, B)
assert np.allclose(C, A @ B)

# Batch matrix multiplication: (batch, m, p) x (batch, p, n) -> (batch, m, n)
A = torch.randn(32, 8, 64)   # 32 batches, 8x64
B = torch.randn(32, 64, 16)  # 32 batches, 64x16
C = torch.einsum('bmp,bpn->bmn', A, B)
assert C.shape == (32, 8, 16)

# Dot product: c = sum_i a_i b_i
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.einsum('i,i->', a, b)  # 32

# Outer product: C_ij = a_i * b_j
C = np.einsum('i,j->ij', a, b)

# Trace: t = sum_i A_ii
A = np.random.randn(4, 4)
t = np.einsum('ii->', A)
assert np.isclose(t, np.trace(A))

# Diagonal: d_i = A_ii
d = np.einsum('ii->i', A)
assert np.allclose(d, np.diag(A))

# Transpose: B_ji = A_ij
B = np.einsum('ij->ji', A)
assert np.allclose(B, A.T)

# Multi-head attention scores (query @ key^T)
# Q: (batch, heads, seq_q, d_k)
# K: (batch, heads, seq_k, d_k)
# scores: (batch, heads, seq_q, seq_k)
Q = torch.randn(8, 12, 128, 64)
K = torch.randn(8, 12, 128, 64)
scores = torch.einsum('bhqd,bhkd->bhqk', Q, K)
assert scores.shape == (8, 12, 128, 128)
```

The power of `einsum` is that it unambiguously specifies the computation without worrying about `transpose`, `reshape`, `unsqueeze`, or `permute` calls. For complex tensor operations in attention mechanisms, convolutions, and graph neural networks, `einsum` is often the clearest expression.

---

## 2.10 Practical Applications

### 2.10.1 PCA Derivation from First Principles

**Principal Component Analysis (PCA)** finds the directions of maximum variance in data. Here we derive it using eigendecomposition.

Given centered data $\mathbf{X} \in \mathbb{R}^{n \times d}$ (each row is a sample, mean subtracted), we seek the direction $\mathbf{w} \in \mathbb{R}^d$ that maximizes the variance of the projected data:

$$\max_{\|\mathbf{w}\|=1} \text{Var}(\mathbf{Xw}) = \max_{\|\mathbf{w}\|=1} \frac{1}{n}\|\mathbf{Xw}\|^2 = \max_{\|\mathbf{w}\|=1} \mathbf{w}^T \left(\frac{1}{n}\mathbf{X}^T\mathbf{X}\right) \mathbf{w}$$

Let $\mathbf{C} = \frac{1}{n}\mathbf{X}^T\mathbf{X}$ be the covariance matrix. We want:

$$\max_{\|\mathbf{w}\|=1} \mathbf{w}^T\mathbf{C}\mathbf{w}$$

Using a Lagrange multiplier for the constraint $\|\mathbf{w}\|^2 = 1$:

$$\mathcal{L} = \mathbf{w}^T\mathbf{C}\mathbf{w} - \lambda(\mathbf{w}^T\mathbf{w} - 1)$$

Setting $\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 0$:

$$2\mathbf{C}\mathbf{w} - 2\lambda\mathbf{w} = 0 \implies \mathbf{C}\mathbf{w} = \lambda\mathbf{w}$$

This is the eigenvector equation. The direction of maximum variance is the eigenvector of $\mathbf{C}$ with the largest eigenvalue. The second principal component is the eigenvector with the second-largest eigenvalue, and so on.

The variance captured by the $i$-th component is $\lambda_i$, and the fraction of total variance explained is $\lambda_i / \sum_j \lambda_j$.

```python
from sklearn.datasets import load_digits

# Load data
digits = load_digits()
X = digits.data  # (1797, 64) — 8x8 pixel images

# Step 1: Center the data
X_centered = X - X.mean(axis=0)

# Step 2: Compute covariance matrix
C = X_centered.T @ X_centered / len(X_centered)  # (64, 64)

# Step 3: Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(C)

# eigh returns in ascending order; we want descending
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Step 4: Project data onto top k components
k = 10
W = eigenvectors[:, :k]        # (64, 10) — projection matrix
X_projected = X_centered @ W   # (1797, 10)

# Variance explained
explained_ratio = eigenvalues[:k] / eigenvalues.sum()
print(f"Top {k} components explain {explained_ratio.sum():.1%} of variance")

# Alternative: SVD-based PCA (more numerically stable)
U, sigma, Vt = np.linalg.svd(X_centered, full_matrices=False)
X_projected_svd = U[:, :k] * sigma[:k]  # Equivalent result
```

### 2.10.2 Word Embeddings and Linear Algebra

Word embeddings (Word2Vec, GloVe) represent words as vectors in $\mathbb{R}^d$ where linear relationships encode semantic meaning (Mikolov et al., 2013):

$$\text{vec}(\text{"king"}) - \text{vec}(\text{"man"}) + \text{vec}(\text{"woman"}) \approx \text{vec}(\text{"queen"})$$

This is vector arithmetic — addition and subtraction in the embedding space. Similarity between words is measured by the cosine of the angle between their vectors (dot product of normalized vectors).

```python
# Using pre-trained GloVe embeddings
import gensim.downloader as api

model = api.load("glove-wiki-gigaword-100")

# Analogy: king - man + woman = ?
result = model.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=3
)
# [('queen', 0.7698), ...]

# Similarity
sim = model.similarity("cat", "dog")  # ~0.80
sim = model.similarity("cat", "car")  # ~0.30

# The embedding matrix itself is a linear algebra object
# Shape: (vocab_size, embedding_dim)
# PCA on embeddings reveals semantic axes
```

### 2.10.3 Linear Algebra in Neural Networks

Every forward pass through a neural network is a sequence of linear algebra operations interspersed with nonlinearities:

```python
import torch
import torch.nn as nn

# A simple feedforward network
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Each Linear layer stores a weight MATRIX and bias VECTOR
        self.fc1 = nn.Linear(input_dim, hidden_dim)   # W1: (hidden, input)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # W2: (output, hidden)

    def forward(self, x):
        # x: (batch, input_dim) — a MATRIX of batch vectors
        h = torch.relu(self.fc1(x))   # h = ReLU(W1 @ x + b1)
        y = self.fc2(h)               # y = W2 @ h + b2
        return y

# The entire computation is:
# y = W2 @ ReLU(W1 @ x + b1) + b2
# Without ReLU, this collapses to y = (W2 @ W1) @ x + (W2 @ b1 + b2)
# = W_combined @ x + b_combined
# A single linear transformation! Nonlinearity is essential.

model = MLP(784, 256, 10)
# Total parameters: 784*256 + 256 + 256*10 + 10 = 203,530
# All are matrix/vector entries
print(sum(p.numel() for p in model.parameters()))  # 203530
```

---

## 2.11 Numerical Considerations

### 2.11.1 Condition Number

The **condition number** of a matrix $\kappa(\mathbf{A}) = \|\mathbf{A}\| \cdot \|\mathbf{A}^{-1}\| = \sigma_{\max} / \sigma_{\min}$ measures how sensitive the solution of $\mathbf{Ax} = \mathbf{b}$ is to perturbations. A large condition number means the problem is ill-conditioned — small errors in $\mathbf{A}$ or $\mathbf{b}$ cause large errors in $\mathbf{x}$.

```python
# Well-conditioned matrix
A_good = np.array([[2, 0], [0, 1]])
print(f"Condition number: {np.linalg.cond(A_good):.1f}")  # 2.0

# Ill-conditioned matrix
A_bad = np.array([[1, 1], [1, 1.0001]])
print(f"Condition number: {np.linalg.cond(A_bad):.1f}")  # ~40000

# Practical implication: never invert ill-conditioned matrices
# Use np.linalg.solve(A, b) instead of np.linalg.inv(A) @ b
```

### 2.11.2 Numerical Stability Tips

1. **Never explicitly compute matrix inverses** when solving linear systems. Use `np.linalg.solve()`.
2. **Use `eigh` instead of `eig`** for symmetric matrices — it is faster and more accurate.
3. **Add regularization** (e.g., $\mathbf{A} + \epsilon\mathbf{I}$) to improve conditioning.
4. **Use SVD-based methods** for rank-deficient problems.
5. **Prefer `float64` over `float32`** when numerical precision matters (but prefer `float32` or `float16` for deep learning speed).

---

## Exercises

### Conceptual

1. Prove that the dot product $\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$ by using the law of cosines applied to the triangle formed by $\mathbf{u}$, $\mathbf{v}$, and $\mathbf{u} - \mathbf{v}$.

2. Show that if $\mathbf{Q}$ is orthogonal, then $\|\mathbf{Qx}\| = \|\mathbf{x}\|$ for all $\mathbf{x}$. Why does this make orthogonal matrices useful for neural network initialization?

3. Explain why PCA finds the directions of maximum variance. Start from the optimization problem and derive the eigenvalue equation.

4. In LoRA fine-tuning, the weight update $\Delta\mathbf{W} = \mathbf{BA}$ has rank at most $r$. If the original weight matrix $\mathbf{W} \in \mathbb{R}^{4096 \times 4096}$ and $r = 16$, how many parameters does LoRA add compared to full fine-tuning? Express as a percentage.

5. Why is the spectral norm (largest singular value) relevant for Lipschitz continuity of neural network layers? How does spectral normalization stabilize GAN training?

### Programming

6. Implement PCA from scratch using (a) eigendecomposition of the covariance matrix and (b) SVD. Verify they give the same results on the Iris dataset.

7. Implement image compression using truncated SVD. For a 512x512 grayscale image, plot reconstruction quality (PSNR) vs. compression ratio for ranks $k = 1, 5, 10, 20, 50, 100, 200$.

8. Using `torch.einsum`, implement the scaled dot-product attention formula: $\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{QK}^T}{\sqrt{d_k}}\right)\mathbf{V}$.

9. Write a function that computes the condition number of the feature matrix $\mathbf{X}^T\mathbf{X}$ for polynomial regression of degree $d$. Show that the condition number grows rapidly with $d$, demonstrating why high-degree polynomial regression is numerically unstable.

10. Build a simple movie recommendation system using SVD. Use the MovieLens 100K dataset, compute the truncated SVD with $k = 50$ latent factors, and predict ratings for unseen user-movie pairs. Report RMSE.

---

## References

- Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 2: Linear Algebra. MIT Press.
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv preprint arXiv:2106.09685*.
- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. *arXiv preprint arXiv:1301.3781*.
- Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2018). Spectral Normalization for Generative Adversarial Networks. *ICLR*.
- Sanderson, G. (2016). *Essence of Linear Algebra* [Video series]. 3Blue1Brown. https://www.3blue1brown.com/topics/linear-algebra
- Saxe, A. M., McClelland, J. L., & Ganguli, S. (2014). Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. *ICLR*.
- Strang, G. (2016). *Introduction to Linear Algebra*, 5th Edition. Wellesley-Cambridge Press.
- Trefethen, L. N., & Bau III, D. (1997). *Numerical Linear Algebra*. SIAM.

---

*Next chapter: Calculus and Optimization — the engine that powers learning in every machine learning algorithm.*
