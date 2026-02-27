# Chapter 5: Classical Machine Learning with Scikit-Learn

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Construct end-to-end machine learning pipelines encompassing data collection, preprocessing, feature engineering, model selection, evaluation, and deployment.
2. Derive and implement the core supervised learning algorithms --- linear regression, logistic regression, support vector machines, decision trees, and ensemble methods --- from their mathematical foundations.
3. Apply advanced gradient boosting frameworks (XGBoost, LightGBM, CatBoost) and understand their algorithmic innovations.
4. Perform unsupervised learning using clustering algorithms (K-Means, DBSCAN, hierarchical, GMMs) and dimensionality reduction techniques (PCA, t-SNE, UMAP).
5. Engineer features systematically through encoding, scaling, imputation, and selection strategies.
6. Evaluate models rigorously using appropriate metrics, cross-validation schemes, calibration techniques, and strategies for handling class imbalance.
7. Optimize hyperparameters using grid search, random search, Bayesian optimization, and Hyperband.
8. Understand when and how to leverage AutoML tools versus manual tuning.

---

## 5.1 The Machine Learning Pipeline

Every machine learning project, regardless of complexity, follows a structured pipeline. Understanding this pipeline is not merely a matter of engineering hygiene --- it is the difference between models that work in notebooks and models that work in production.

### 5.1.1 Data Collection and Ingestion

The pipeline begins with data. In practice, data arrives from databases (SQL and NoSQL), APIs, log streams, flat files, web scraping, and increasingly from real-time event systems. The quality of downstream models is bounded by the quality and representativeness of this data --- a principle sometimes stated as "garbage in, garbage out," but more precisely formalized by the notion of dataset shift (Quionero-Candela et al., 2009).

Key considerations during data collection include:

- **Sampling bias**: Does the collected data reflect the distribution the model will encounter at inference time?
- **Label quality**: For supervised learning, label noise directly degrades model performance (Frenay & Verleysen, 2014).
- **Data freshness**: Temporal drift between training and serving data is a leading cause of model degradation.

### 5.1.2 Preprocessing

Raw data is rarely suitable for direct consumption by learning algorithms. Preprocessing involves:

- **Missing value treatment**: Deletion (listwise or pairwise), imputation (mean, median, mode, KNN, iterative), or indicator-based approaches.
- **Outlier detection and handling**: Statistical methods (z-score, IQR), domain-informed clipping, or robust estimators.
- **Type conversion**: Ensuring numerical features are numeric, dates are parsed, and categorical variables are properly typed.

### 5.1.3 Feature Engineering

Feature engineering is the process of transforming raw variables into representations that better expose the underlying structure to learning algorithms. We devote Section 5.10 to this topic in depth.

### 5.1.4 Model Selection

Model selection involves choosing both the algorithm family and the specific configuration. The No Free Lunch theorem (Wolpert, 1996) tells us that no single algorithm dominates across all possible problems. In practice, structured tabular data often favors gradient-boosted decision trees, while unstructured data (images, text) favors neural networks.

### 5.1.5 Evaluation

A model is only as good as our ability to measure its performance honestly. We discuss evaluation in depth in Sections 5.11 and 5.12.

### 5.1.6 Deployment

Deployment bridges the gap between experimentation and value delivery. We cover MLOps and deployment in detail in later chapters, but the key principle here is that the pipeline must be reproducible: every transformation applied during training must be applied identically during inference.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# A minimal but production-ready pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000))
])

# The pipeline ensures the same transformations at train and test time
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

---

## 5.2 Linear Regression

Linear regression is the foundation upon which much of statistical learning is built. Despite its simplicity, it embodies principles --- loss minimization, regularization, the bias-variance tradeoff --- that recur throughout machine learning.

### 5.2.1 The Model

Given a dataset of $n$ observations with $p$ features, we model the relationship between input $\mathbf{x} \in \mathbb{R}^p$ and output $y \in \mathbb{R}$ as:

$$y = \mathbf{w}^T \mathbf{x} + b + \epsilon$$

where $\mathbf{w} \in \mathbb{R}^p$ is the weight vector, $b$ is the bias (intercept), and $\epsilon$ is irreducible error. For notational convenience, we absorb the bias into $\mathbf{w}$ by prepending a 1 to each input vector, yielding $y = \mathbf{w}^T \mathbf{x}$.

### 5.2.2 Ordinary Least Squares: The Normal Equation

We seek weights that minimize the sum of squared residuals:

$$\mathcal{L}(\mathbf{w}) = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 = (\mathbf{y} - \mathbf{X}\mathbf{w})^T(\mathbf{y} - \mathbf{X}\mathbf{w})$$

Expanding:

$$\mathcal{L}(\mathbf{w}) = \mathbf{y}^T\mathbf{y} - 2\mathbf{w}^T\mathbf{X}^T\mathbf{y} + \mathbf{w}^T\mathbf{X}^T\mathbf{X}\mathbf{w}$$

Taking the gradient with respect to $\mathbf{w}$ and setting it to zero:

$$\nabla_{\mathbf{w}} \mathcal{L} = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\mathbf{w} = \mathbf{0}$$

Solving:

$$\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

This is the **normal equation**. It has a closed-form solution but requires inverting a $p \times p$ matrix, which costs $O(p^3)$ and is numerically unstable when $\mathbf{X}^T\mathbf{X}$ is ill-conditioned or singular. In practice, scikit-learn uses the pseudoinverse via SVD decomposition rather than direct inversion (Hastie et al., 2009).

### 5.2.3 Gradient Descent Solution

When $n$ or $p$ is large, iterative optimization via gradient descent is preferred. The normal equation requires $O(p^3)$ computation for the matrix inversion, which becomes prohibitive when $p$ exceeds a few thousand. Furthermore, for very large $n$, even forming $\mathbf{X}^T\mathbf{X}$ (which is $O(np^2)$) may be impractical. Gradient descent provides a scalable alternative.

The update rule is:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w}_t)$$

where $\eta > 0$ is the learning rate. For the MSE loss, the gradient is:

$$\nabla_{\mathbf{w}} \mathcal{L} = -\frac{2}{n} \mathbf{X}^T(\mathbf{y} - \mathbf{X}\mathbf{w})$$

The learning rate $\eta$ is critical: too large and the algorithm diverges; too small and convergence is impractically slow. For convex problems like linear regression, any $\eta < 2/\lambda_{\max}$ (where $\lambda_{\max}$ is the largest eigenvalue of $\mathbf{X}^T\mathbf{X}/n$) guarantees convergence, but the optimal rate is $\eta = 2/(\lambda_{\max} + \lambda_{\min})$, yielding convergence in $O(\kappa \log(1/\epsilon))$ iterations where $\kappa = \lambda_{\max}/\lambda_{\min}$ is the condition number.

Variants include:

- **Batch gradient descent**: Uses all $n$ samples per update. Stable but slow for large $n$. Each iteration costs $O(np)$.
- **Stochastic gradient descent (SGD)**: Uses one sample per update. Noisy but fast. Each iteration costs $O(p)$. The noisy gradients can actually help escape saddle points in non-convex optimization.
- **Mini-batch SGD**: Uses a batch of $B$ samples. The practical standard, as it balances computational efficiency (parallelism on GPUs) with gradient quality. Typical batch sizes range from 32 to 512.

The convergence rate of SGD is $O(1/\sqrt{T})$ for convex functions and $O(1/T)$ for strongly convex functions (with appropriate learning rate decay), compared to $O(\rho^T)$ for batch gradient descent where $\rho < 1$ depends on the condition number (Bottou et al., 2018).

### 5.2.4 Ridge Regression (L2 Regularization)

When features are correlated (multicollinearity) or when $p > n$, the OLS solution is unstable. Ridge regression adds an L2 penalty:

$$\mathcal{L}_{\text{Ridge}}(\mathbf{w}) = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \alpha \|\mathbf{w}\|_2^2$$

The closed-form solution becomes:

$$\mathbf{w}^*_{\text{Ridge}} = (\mathbf{X}^T\mathbf{X} + \alpha \mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$

The addition of $\alpha \mathbf{I}$ ensures the matrix is always invertible. Geometrically, Ridge regression shrinks weights toward zero but never exactly to zero (Hoerl & Kennard, 1970).

### 5.2.5 Lasso Regression (L1 Regularization)

Lasso replaces the L2 penalty with an L1 penalty:

$$\mathcal{L}_{\text{Lasso}}(\mathbf{w}) = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \alpha \|\mathbf{w}\|_1$$

The L1 penalty induces sparsity --- some weights become exactly zero, effectively performing feature selection. There is no closed-form solution; Lasso is solved via coordinate descent (Tibshirani, 1996).

### 5.2.6 ElasticNet

ElasticNet combines L1 and L2 penalties:

$$\mathcal{L}_{\text{ElasticNet}}(\mathbf{w}) = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \alpha \rho \|\mathbf{w}\|_1 + \frac{\alpha(1 - \rho)}{2} \|\mathbf{w}\|_2^2$$

where $\rho \in [0, 1]$ controls the mix. ElasticNet inherits sparsity from Lasso and stability from Ridge, and is especially useful when features are grouped and correlated (Zou & Hastie, 2005).

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=50, n_informative=10,
                       noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42)

models = {
    'OLS': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    n_nonzero = np.sum(model.coef_ != 0)
    print(f"{name:12s} | RMSE: {rmse:.2f} | Non-zero coefficients: {n_nonzero}")
```

---

## 5.3 Logistic Regression

Despite its name, logistic regression is a classification algorithm. It models the probability that an input belongs to a particular class using the logistic (sigmoid) function.

### 5.3.1 The Sigmoid Function

For binary classification with classes $\{0, 1\}$, we model:

$$P(y = 1 | \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x}}}$$

The sigmoid function $\sigma(z) = 1/(1 + e^{-z})$ maps any real number to $(0, 1)$. Its derivative has a convenient form: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.

### 5.3.2 Maximum Likelihood Estimation

We derive the loss function from first principles using maximum likelihood. Given $n$ independent observations, the likelihood is:

$$\mathcal{L}(\mathbf{w}) = \prod_{i=1}^{n} p_i^{y_i} (1 - p_i)^{1 - y_i}$$

where $p_i = \sigma(\mathbf{w}^T \mathbf{x}_i)$. Taking the negative log-likelihood (which we minimize):

$$\text{NLL}(\mathbf{w}) = -\sum_{i=1}^{n} \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right]$$

This is the **binary cross-entropy loss**. The gradient is:

$$\nabla_{\mathbf{w}} \text{NLL} = \sum_{i=1}^{n} (p_i - y_i) \mathbf{x}_i = \mathbf{X}^T(\mathbf{p} - \mathbf{y})$$

This has no closed-form solution. We solve it iteratively using gradient descent or Newton's method (Iteratively Reweighted Least Squares, IRLS).

### 5.3.3 Regularization

Just as with linear regression, regularization prevents overfitting:

- **L2 (default in scikit-learn)**: $\text{NLL} + \frac{1}{2C}\|\mathbf{w}\|_2^2$
- **L1**: $\text{NLL} + \frac{1}{C}\|\mathbf{w}\|_1$
- **ElasticNet**: A combination of both.

Note that scikit-learn parameterizes the regularization strength as $C = 1/\alpha$: larger $C$ means less regularization.

### 5.3.4 Multi-Class: Softmax Regression

For $K > 2$ classes, logistic regression generalizes to **softmax regression**. Each class $k$ has its own weight vector $\mathbf{w}_k$, and the probability of class $k$ is:

$$P(y = k | \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T \mathbf{x}}}$$

The loss becomes the **categorical cross-entropy**:

$$\mathcal{L} = -\sum_{i=1}^{n} \sum_{k=1}^{K} \mathbb{1}[y_i = k] \log P(y_i = k | \mathbf{x}_i)$$

Scikit-learn supports multi-class logistic regression via `multi_class='multinomial'` with solvers like `lbfgs`, `newton-cg`, or `saga`.

### 5.3.5 Solvers in Scikit-Learn

| Solver | Regularization | Multi-class | Large datasets |
|--------|---------------|-------------|----------------|
| `lbfgs` | L2, none | Multinomial | Moderate |
| `liblinear` | L1, L2 | OVR only | Moderate |
| `newton-cg` | L2, none | Multinomial | Moderate |
| `sag` | L2, none | Multinomial | Large |
| `saga` | L1, L2, ElasticNet | Multinomial | Large |

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

X, y = load_iris(return_X_y=True)

# Multinomial logistic regression with L2 regularization
model = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    multi_class='multinomial',
    max_iter=200
)

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

---

## 5.4 Support Vector Machines

Support Vector Machines (SVMs) represent one of the most elegant formulations in machine learning, combining geometric intuition with powerful mathematical machinery from optimization theory (Cortes & Vapnik, 1995).

### 5.4.1 Maximum Margin Classifier

For linearly separable data with labels $y_i \in \{-1, +1\}$, a linear classifier defines a decision boundary $\mathbf{w}^T \mathbf{x} + b = 0$. The **margin** is the distance between the decision boundary and the nearest data point. The SVM finds the hyperplane that maximizes this margin.

The distance from a point $\mathbf{x}_i$ to the hyperplane is $\frac{y_i(\mathbf{w}^T \mathbf{x}_i + b)}{\|\mathbf{w}\|}$. The margin is $\frac{2}{\|\mathbf{w}\|}$ (the minimum distance from either class, doubled). Maximizing the margin is equivalent to minimizing $\|\mathbf{w}\|^2$ subject to the constraint that all points are correctly classified with margin at least 1:

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{subject to} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \quad \forall i$$

### 5.4.2 Soft Margin SVM

Real data is rarely linearly separable. The soft margin formulation introduces slack variables $\xi_i \geq 0$ that allow some points to violate the margin:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i$$

$$\text{subject to} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

The parameter $C$ controls the tradeoff between maximizing the margin and minimizing classification errors. Large $C$ penalizes misclassifications heavily (low bias, high variance); small $C$ allows more violations (high bias, low variance).

### 5.4.3 The Dual Formulation

Using Lagrange multipliers $\alpha_i \geq 0$, the dual problem is:

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i=1}^{n}\sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j$$

$$\text{subject to} \quad 0 \leq \alpha_i \leq C, \quad \sum_{i=1}^{n} \alpha_i y_i = 0$$

The key insight is that the solution depends only on inner products $\mathbf{x}_i^T \mathbf{x}_j$. Points with $\alpha_i > 0$ are the **support vectors** --- they lie on or within the margin and fully determine the decision boundary.

### 5.4.4 The Kernel Trick

The dual formulation's dependence on inner products enables the **kernel trick**: we replace $\mathbf{x}_i^T \mathbf{x}_j$ with a kernel function $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j)$, where $\phi$ maps inputs to a high-dimensional (possibly infinite-dimensional) feature space --- without ever computing $\phi$ explicitly.

Common kernels include:

- **Linear**: $K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j$
- **Polynomial**: $K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i^T \mathbf{x}_j + r)^d$
- **Radial Basis Function (RBF)**: $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$

The RBF kernel maps to an infinite-dimensional space and is the default in scikit-learn. The parameter $\gamma$ controls the radius of influence of each support vector: large $\gamma$ means tight, local influence (potential overfitting); small $\gamma$ means broad, smooth influence (potential underfitting).

```python
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Non-linearly separable data
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
X = StandardScaler().fit_transform(X)

# Grid search over kernel parameters
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.01, 0.1, 1],
    'kernel': ['rbf']
}

svm = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
svm.fit(X, y)
print(f"Best params: {svm.best_params_}")
print(f"Best accuracy: {svm.best_score_:.3f}")
```

---

## 5.5 Decision Trees

Decision trees partition the feature space into rectangular regions via a sequence of binary splits, making predictions based on the majority class (classification) or mean value (regression) within each region (Breiman et al., 1984).

### 5.5.1 Splitting Criteria

At each node, the tree selects the feature and threshold that best separate the data. The quality of a split is measured by an **impurity function**.

**Gini Impurity** (used by CART):

$$G(t) = 1 - \sum_{k=1}^{K} p_k^2$$

where $p_k$ is the proportion of class $k$ at node $t$. Gini impurity reaches its minimum (0) when a node is pure (all samples belong to one class).

**Information Gain / Entropy** (used by ID3, C4.5):

$$H(t) = -\sum_{k=1}^{K} p_k \log_2 p_k$$

The information gain from a split is:

$$\text{IG}(t, \text{split}) = H(t) - \sum_{c \in \{\text{left}, \text{right}\}} \frac{n_c}{n_t} H(c)$$

C4.5 improves upon ID3 by using the **gain ratio**, which normalizes information gain by the intrinsic information of the split, reducing bias toward features with many distinct values (Quinlan, 1993).

### 5.5.2 Tree Algorithms

- **ID3**: Uses information gain. Handles only categorical features. No pruning.
- **C4.5**: Uses gain ratio. Handles continuous features via thresholding. Implements error-based pruning.
- **CART** (Classification and Regression Trees): Uses Gini impurity (classification) or MSE (regression). Produces binary trees only. Scikit-learn implements CART.

### 5.5.3 Regression Trees

For regression, CART uses variance reduction (or equivalently, MSE minimization) as the splitting criterion. At each node, the prediction is the mean of the target values, and the best split minimizes:

$$\text{MSE}_{\text{split}} = \frac{n_L}{n} \text{Var}(y_L) + \frac{n_R}{n} \text{Var}(y_R)$$

where $y_L$ and $y_R$ are the target values in the left and right child nodes, respectively. The final prediction for a leaf node containing samples $S$ is simply $\hat{y} = \frac{1}{|S|}\sum_{i \in S} y_i$.

### 5.5.4 Pruning Strategies

Unpruned trees overfit --- they memorize the training data. A fully grown tree will have one leaf per training sample, achieving zero training error but poor generalization. Pruning methods include:

- **Pre-pruning (early stopping)**: Limit `max_depth`, `min_samples_split`, `min_samples_leaf`, or `max_leaf_nodes`. These constraints prevent the tree from growing beyond a certain complexity. The disadvantage is that they may stop splitting too early, missing useful interactions that would only appear in deeper branches.
- **Post-pruning (cost-complexity pruning)**: Grow the full tree, then prune subtrees that do not improve performance on a validation set. CART uses **minimal cost-complexity pruning**, which minimizes $R_\alpha(T) = R(T) + \alpha |T|$, where $R(T)$ is the misclassification rate (or MSE), $|T|$ is the number of terminal nodes, and $\alpha \geq 0$ is the complexity parameter. As $\alpha$ increases, more subtrees are pruned. This is controlled by the parameter `ccp_alpha` in scikit-learn.

The advantage of post-pruning is that it allows the tree to discover complex interactions first, then removes those that do not generalize. Empirically, post-pruning almost always outperforms pre-pruning alone, though combining both strategies is common in practice (Breiman et al., 1984).

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score

X, y = load_wine(return_X_y=True)

# Cost-complexity pruning path
tree = DecisionTreeClassifier(random_state=42)
path = tree.cost_complexity_pruning_path(X, y)
ccp_alphas = path.ccp_alphas

# Find the best alpha via cross-validation
best_score, best_alpha = 0, 0
for alpha in ccp_alphas:
    dt = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    scores = cross_val_score(dt, X, y, cv=5)
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_alpha = alpha

print(f"Best ccp_alpha: {best_alpha:.4f}, CV accuracy: {best_score:.3f}")
```

---

## 5.6 Ensemble Methods

Ensemble methods combine multiple base learners to produce a model that is more accurate than any individual member. The theoretical justification comes from the bias-variance decomposition: ensembles can reduce variance (bagging), reduce bias (boosting), or both (stacking).

### 5.6.1 Bagging and Random Forests

**Bootstrap Aggregating (Bagging)** trains multiple base learners on bootstrap samples (random samples with replacement) of the training data, then averages their predictions (regression) or takes a majority vote (classification) (Breiman, 1996).

**Random Forests** extend bagging by adding feature randomization: at each split, only a random subset of $m$ features (typically $m = \sqrt{p}$ for classification, $m = p/3$ for regression) is considered. This decorrelates the trees, further reducing variance (Breiman, 2001).

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score

X, y = load_breast_cancer(return_X_y=True)

rf = RandomForestClassifier(
    n_estimators=500,
    max_features='sqrt',
    max_depth=None,
    min_samples_leaf=2,
    oob_score=True,      # Out-of-bag estimate
    n_jobs=-1,
    random_state=42
)
rf.fit(X, y)
print(f"OOB accuracy: {rf.oob_score_:.3f}")

# Feature importance (mean decrease in impurity)
importances = rf.feature_importances_
```

### 5.6.2 Boosting

Boosting builds an ensemble sequentially, where each new model focuses on the errors of the previous ensemble. While bagging reduces variance (by averaging independent noisy predictions), boosting primarily reduces bias (by iteratively correcting systematic errors).

**AdaBoost** (Freund & Schapire, 1997) maintains sample weights $w_i$ initialized uniformly. At each iteration $t$:

1. Train a weak learner $h_t$ on the weighted data.
2. Compute its weighted error: $\epsilon_t = \sum_{i: h_t(x_i) \neq y_i} w_i / \sum_i w_i$.
3. Compute the learner's weight: $\alpha_t = \frac{1}{2}\ln\frac{1 - \epsilon_t}{\epsilon_t}$.
4. Update sample weights: $w_i \leftarrow w_i \exp(-\alpha_t y_i h_t(x_i))$, then renormalize.

Misclassified samples receive exponentially higher weight, forcing subsequent learners to focus on hard examples. The final prediction is $H(\mathbf{x}) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(\mathbf{x})\right)$. AdaBoost can be shown to minimize exponential loss, and its training error decreases exponentially with the number of iterations (under mild assumptions on the weak learners).

**Gradient Boosting** (Friedman, 2001) generalizes boosting to arbitrary differentiable loss functions. At each step $m$, a new base learner $h_m$ is fit to the negative gradient (pseudo-residuals) of the loss:

$$r_{im} = -\left[\frac{\partial \mathcal{L}(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)}\right]_{F = F_{m-1}}$$

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})$$

where $\eta$ is the learning rate (shrinkage).

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,        # Stochastic gradient boosting
    random_state=42
)
scores = cross_val_score(gb, X, y, cv=5, scoring='accuracy')
print(f"Gradient Boosting accuracy: {scores.mean():.3f}")
```

### 5.6.3 Stacking and Blending

**Stacking** trains a meta-learner on the out-of-fold predictions of multiple base learners. The base learners are diverse models (e.g., Random Forest, SVM, Logistic Regression), and the meta-learner learns how to optimally combine them (Wolpert, 1992).

**Blending** is a simplified variant that uses a single hold-out set for generating meta-features instead of cross-validation.

```python
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svc', SVC(kernel='rbf', probability=True, random_state=42)),
    ('dt', DecisionTreeClassifier(max_depth=5, random_state=42))
]

stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)
scores = cross_val_score(stacking, X, y, cv=5, scoring='accuracy')
print(f"Stacking accuracy: {scores.mean():.3f}")
```

---

## 5.7 XGBoost, LightGBM, and CatBoost

While scikit-learn's `GradientBoostingClassifier` is instructive, production-grade gradient boosting is dominated by three frameworks that introduce algorithmic innovations for speed, accuracy, and usability.

### 5.7.1 XGBoost

XGBoost (eXtreme Gradient Boosting) was introduced by Chen & Guestrin (2016) and rapidly became the dominant algorithm for structured data competitions.

**Second-Order Taylor Expansion.** Rather than fitting trees to pseudo-residuals alone, XGBoost uses a second-order Taylor approximation of the loss. At iteration $t$, the objective for adding tree $f_t$ is:

$$\mathcal{L}^{(t)} \approx \sum_{i=1}^{n} \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t(\mathbf{x}_i)^2 \right] + \Omega(f_t)$$

where $g_i = \partial \mathcal{L}/\partial \hat{y}_i^{(t-1)}$ is the first-order gradient, $h_i = \partial^2 \mathcal{L}/\partial (\hat{y}_i^{(t-1)})^2$ is the second-order gradient (Hessian), and $\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$ is a regularization term on the tree complexity (where $T$ is the number of leaves and $w_j$ are the leaf weights).

The optimal weight for leaf $j$ and the corresponding gain for a split are:

$$w_j^* = -\frac{G_j}{H_j + \lambda}, \quad \text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma$$

where $G_j = \sum_{i \in I_j} g_i$ and $H_j = \sum_{i \in I_j} h_i$.

**Handling Missing Values.** XGBoost learns the optimal direction for missing values at each split during training, eliminating the need for imputation.

```python
import xgboost as xgb
from sklearn.model_selection import cross_val_score

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,       # L1 regularization
    'reg_lambda': 1.0,      # L2 regularization
    'min_child_weight': 5,
    'tree_method': 'hist'   # Histogram-based splitting
}

model = xgb.train(
    params, dtrain,
    num_boost_round=500,
    evals=[(dtrain, 'train'), (dtest, 'eval')],
    early_stopping_rounds=50,
    verbose_eval=100
)
```

### 5.7.2 LightGBM

LightGBM (Ke et al., 2017) introduces two key innovations for speed:

**Gradient-based One-Side Sampling (GOSS).** Instead of using all data to compute information gain, GOSS keeps all instances with large gradients (which contribute more to information gain) and randomly samples from instances with small gradients, applying a correction factor.

**Exclusive Feature Bundling (EFB).** Features that are mutually exclusive (rarely non-zero simultaneously) are bundled into a single feature, reducing the effective number of features.

**Leaf-wise Growth.** Unlike XGBoost's level-wise (depth-first) growth, LightGBM grows trees leaf-wise: it always splits the leaf with the highest gain reduction. This produces deeper, more asymmetric trees that can achieve lower loss with fewer iterations, but risks overfitting --- controlled by `num_leaves` and `max_depth`.

```python
import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

model = lgb.train(
    params, lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_eval],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)
```

### 5.7.3 CatBoost

CatBoost (Prokhorenkova et al., 2018) was designed to handle categorical features natively.

**Ordered Target Encoding.** Naive target encoding (replacing a category with the mean target value) causes target leakage. CatBoost uses **ordered target statistics**: for each sample, the target statistic is computed using only samples that precede it in a random permutation. This is formally equivalent to a type of online learning and avoids overfitting on rare categories.

**Symmetric (Oblivious) Trees.** CatBoost builds symmetric trees where the same split condition is used across all nodes at a given depth. This acts as a strong regularizer and enables extremely fast inference via bitwise operations.

```python
from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    cat_features=[0, 3, 7],  # Indices of categorical columns
    eval_metric='AUC',
    early_stopping_rounds=50,
    verbose=100
)
cat_model.fit(X_train, y_train, eval_set=(X_test, y_test))
```

---

## 5.8 Clustering

Clustering is the task of grouping unlabeled data points such that points within a group are more similar to each other than to points in other groups. Unlike classification, there is no ground truth; the quality of clustering depends on the chosen similarity measure and the intended application.

### 5.8.1 K-Means

**Lloyd's Algorithm.** K-Means (Lloyd, 1982) alternates between two steps:

1. **Assignment**: Assign each point to the nearest centroid: $c_i = \arg\min_k \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$
2. **Update**: Recompute centroids as the mean of assigned points: $\boldsymbol{\mu}_k = \frac{1}{|C_k|}\sum_{i \in C_k} \mathbf{x}_i$

The algorithm converges to a local minimum of the within-cluster sum of squares (inertia):

$$\text{WCSS} = \sum_{k=1}^{K} \sum_{i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

**K-Means++ Initialization.** Random initialization can lead to poor local optima. K-Means++ (Arthur & Vassilvitskii, 2007) selects initial centroids with probability proportional to their squared distance from the nearest existing centroid, ensuring well-spread initial centers. This is the default in scikit-learn.

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Elbow method and silhouette analysis
inertias, silhouettes = [], []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, km.labels_))
```

### 5.8.2 DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) discovers clusters of arbitrary shape based on point density (Ester et al., 1996). It requires two parameters:

- **eps**: The radius of the neighborhood around each point.
- **min_samples**: The minimum number of points required in a neighborhood for a point to be a core point.

Points are classified as:
- **Core points**: Have at least `min_samples` points within `eps`.
- **Border points**: Within `eps` of a core point but not core points themselves.
- **Noise points**: Neither core nor border.

DBSCAN does not require specifying the number of clusters, handles noise naturally, and discovers arbitrarily shaped clusters. Its weakness is sensitivity to the `eps` parameter and difficulty with clusters of varying density.

### 5.8.3 Hierarchical Clustering

Agglomerative hierarchical clustering builds a hierarchy of clusters bottom-up:

1. Start with each point as its own cluster.
2. Merge the two closest clusters.
3. Repeat until a single cluster remains.

The linkage criterion defines "closest":
- **Single linkage**: Minimum distance between any two points across clusters.
- **Complete linkage**: Maximum distance.
- **Average linkage**: Average distance.
- **Ward's method**: Minimizes the total within-cluster variance (equivalent to the increase in WCSS upon merging).

The resulting hierarchy is visualized as a **dendrogram**, and the number of clusters is chosen by cutting at a particular height.

### 5.8.4 Gaussian Mixture Models (GMMs)

GMMs model the data as a mixture of $K$ multivariate Gaussian distributions:

$$p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

where $\pi_k$ are mixing coefficients ($\sum_k \pi_k = 1$), $\boldsymbol{\mu}_k$ are means, and $\boldsymbol{\Sigma}_k$ are covariance matrices.

**Expectation-Maximization (EM) Algorithm.** GMMs are fitted using EM (Dempster et al., 1977):

- **E-step**: Compute responsibilities --- the posterior probability that each point belongs to each component:
$$\gamma_{ik} = \frac{\pi_k \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$

- **M-step**: Update parameters using responsibilities:
$$\boldsymbol{\mu}_k = \frac{\sum_i \gamma_{ik} \mathbf{x}_i}{\sum_i \gamma_{ik}}, \quad \boldsymbol{\Sigma}_k = \frac{\sum_i \gamma_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T}{\sum_i \gamma_{ik}}$$

$$\pi_k = \frac{\sum_i \gamma_{ik}}{n}$$

Unlike K-Means, GMMs provide soft cluster assignments (probabilities) and can model elliptical clusters.

```python
from sklearn.mixture import GaussianMixture
import numpy as np

# Fit a GMM and use BIC for model selection
bics = []
for k in range(1, 10):
    gmm = GaussianMixture(n_components=k, covariance_type='full',
                          random_state=42)
    gmm.fit(X)
    bics.append(gmm.bic(X))

optimal_k = np.argmin(bics) + 1
print(f"Optimal number of components (BIC): {optimal_k}")
```

---

## 5.9 Dimensionality Reduction

High-dimensional data suffers from the curse of dimensionality: distances become less meaningful, models require more data, and visualization is impossible. Dimensionality reduction addresses this by projecting data into a lower-dimensional space that preserves important structure.

### 5.9.1 Principal Component Analysis (PCA)

PCA finds the directions of maximum variance in the data and projects onto them (Pearson, 1901; Hotelling, 1933).

**Derivation via eigendecomposition.** Given centered data $\mathbf{X}$ (zero mean), the covariance matrix is:

$$\mathbf{C} = \frac{1}{n-1}\mathbf{X}^T\mathbf{X}$$

We seek the direction $\mathbf{v}$ that maximizes the variance of the projection:

$$\max_{\mathbf{v}} \mathbf{v}^T \mathbf{C} \mathbf{v} \quad \text{subject to} \quad \|\mathbf{v}\| = 1$$

Using Lagrange multipliers, this yields the eigenvalue problem:

$$\mathbf{C}\mathbf{v} = \lambda \mathbf{v}$$

The principal components are the eigenvectors of $\mathbf{C}$, ordered by decreasing eigenvalue $\lambda_1 \geq \lambda_2 \geq \cdots$. The eigenvalue $\lambda_k$ equals the variance explained by the $k$-th component. The fraction of variance explained by the first $d$ components is $\sum_{k=1}^{d} \lambda_k / \sum_{k=1}^{p} \lambda_k$.

In practice, PCA is computed via SVD of $\mathbf{X}$ rather than eigendecomposition of $\mathbf{C}$, which is numerically more stable and efficient.

```python
from sklearn.decomposition import PCA
import numpy as np

pca = PCA(n_components=0.95)  # Retain 95% of variance
X_reduced = pca.fit_transform(X)

print(f"Original dimensions: {X.shape[1]}")
print(f"Reduced dimensions: {X_reduced.shape[1]}")
print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
```

### 5.9.2 t-SNE

t-Distributed Stochastic Neighbor Embedding (t-SNE) (van der Maaten & Hinton, 2008) is a nonlinear technique designed for 2D/3D visualization.

**Pairwise probabilities in high-dimensional space.** For each pair of points, t-SNE computes a conditional probability proportional to a Gaussian similarity:

$$p_{j|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)}$$

The bandwidth $\sigma_i$ is set by the user-specified **perplexity** (a soft measure of the effective number of neighbors). The joint distribution is symmetrized: $p_{ij} = (p_{j|i} + p_{i|j}) / 2n$.

**Pairwise probabilities in low-dimensional space.** The low-dimensional map uses a Student's t-distribution with one degree of freedom:

$$q_{ij} = \frac{(1 + \|\mathbf{z}_i - \mathbf{z}_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|\mathbf{z}_k - \mathbf{z}_l\|^2)^{-1}}$$

The heavy tails of the t-distribution allow moderate distances in the high-dimensional space to be represented as large distances in the map, alleviating the "crowding problem."

**Optimization.** t-SNE minimizes the KL divergence between $P$ and $Q$:

$$\text{KL}(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

This is optimized via gradient descent with momentum and early exaggeration.

### 5.9.3 UMAP

Uniform Manifold Approximation and Projection (UMAP) (McInnes et al., 2018) offers faster computation and better preservation of global structure than t-SNE.

**Theoretical foundation.** UMAP is grounded in Riemannian geometry and algebraic topology. It models the data as lying on a locally connected Riemannian manifold and constructs a fuzzy simplicial set (a weighted graph) that approximates the topological structure. The key steps are:

1. Construct a weighted k-nearest-neighbor graph in high-dimensional space, where edge weights decay with distance (using a locally adaptive exponential kernel).
2. Construct a similar weighted graph in the low-dimensional embedding.
3. Minimize the cross-entropy between the two graph structures using stochastic gradient descent.

In practice, UMAP is significantly faster than t-SNE (the complexity is approximately $O(n \log n)$ compared to t-SNE's $O(n^2)$ for naive implementations) and produces embeddings that better preserve global distances and cluster relationships.

```python
from sklearn.manifold import TSNE
# UMAP requires: pip install umap-learn
import umap

# t-SNE embedding
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X)

# UMAP embedding
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                    random_state=42)
X_umap = reducer.fit_transform(X)
```

---

## 5.10 Feature Engineering

Feature engineering is the process of using domain knowledge to create, transform, and select features that make machine learning algorithms work better. Despite the rise of automated feature learning in deep learning, feature engineering remains critical for tabular data (Zheng & Casari, 2018).

### 5.10.1 Encoding Categorical Variables

**One-Hot Encoding.** Creates binary columns for each category. Suitable for nominal (unordered) categories with moderate cardinality. Avoid for high-cardinality features (e.g., ZIP codes with thousands of values) as it creates a sparse, high-dimensional representation.

**Ordinal Encoding.** Maps categories to integers. Appropriate when categories have a natural order (e.g., education levels: high school < bachelor's < master's < PhD).

**Target Encoding.** Replaces each category with the mean of the target variable for that category. Powerful but prone to overfitting, especially for rare categories. Must be computed using cross-validation or regularization (smoothing) to avoid target leakage.

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder

# One-Hot Encoding
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_ohe = ohe.fit_transform(X_categorical)

# Target Encoding with smoothing
te = TargetEncoder(smoothing=10)
X_te = te.fit_transform(X_categorical, y)
```

### 5.10.2 Scaling

- **StandardScaler**: $x' = (x - \mu) / \sigma$. Centers data at zero with unit variance. Sensitive to outliers.
- **MinMaxScaler**: $x' = (x - x_{\min}) / (x_{\max} - x_{\min})$. Scales to $[0, 1]$. Very sensitive to outliers.
- **RobustScaler**: Uses median and IQR: $x' = (x - \text{median}) / \text{IQR}$. Robust to outliers.

Scaling is critical for algorithms sensitive to feature magnitudes: SVMs, k-NN, logistic regression, PCA, neural networks. Tree-based methods are invariant to monotonic transformations and do not require scaling.

### 5.10.3 Imputation

- **Simple imputation**: Replace missing values with mean, median, or mode. Fast but ignores feature correlations.
- **KNN imputation**: Imputes using the mean of k-nearest neighbors. Preserves local structure.
- **Iterative imputation** (MICE): Models each feature with missing values as a function of other features, iterating until convergence (van Buuren & Groothuis-Oudshoorn, 2011). Equivalent to multiple imputation by chained equations.

```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Iterative imputation (MICE)
imputer = IterativeImputer(max_iter=10, random_state=42)
X_imputed = imputer.fit_transform(X_with_missing)
```

### 5.10.4 Polynomial Features and Interaction Terms

For linear models, polynomial features and interactions allow capturing nonlinear relationships:

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=False,
                          include_bias=False)
X_poly = poly.fit_transform(X)
# For 3 features: creates x1, x2, x3, x1^2, x1*x2, x1*x3, x2^2, x2*x3, x3^2
```

### 5.10.5 Feature Selection

Feature selection reduces dimensionality by removing irrelevant or redundant features:

- **Filter methods**: Statistical tests (chi-squared, ANOVA F-test, mutual information) rank features independently of any model.
- **Wrapper methods**: Evaluate feature subsets using a model (e.g., recursive feature elimination, RFE).
- **Embedded methods**: Feature selection is part of the learning algorithm (e.g., Lasso, tree-based feature importances).

```python
from sklearn.feature_selection import (SelectKBest, mutual_info_classif,
                                       RFE)
from sklearn.ensemble import RandomForestClassifier

# Filter: mutual information
selector = SelectKBest(mutual_info_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Wrapper: Recursive Feature Elimination
rfe = RFE(RandomForestClassifier(n_estimators=100, random_state=42),
          n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y)
```

---

## 5.11 Model Evaluation

Rigorous model evaluation is the foundation of trustworthy machine learning. Selecting the right metric, using appropriate validation strategies, and understanding the bias-variance tradeoff are essential skills (Hastie et al., 2009).

### 5.11.1 Classification Metrics

**Confusion Matrix.** For binary classification, the confusion matrix contains four counts: True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).

From these, we derive:

- **Accuracy**: $(TP + TN) / (TP + TN + FP + FN)$. Misleading when classes are imbalanced.
- **Precision**: $TP / (TP + FP)$. Of all positive predictions, how many are correct?
- **Recall (Sensitivity)**: $TP / (TP + FN)$. Of all actual positives, how many are detected?
- **F1 Score**: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$. The harmonic mean of precision and recall.
- **Specificity**: $TN / (TN + FP)$. Of all actual negatives, how many are correctly identified?

**AUC-ROC.** The Receiver Operating Characteristic curve plots True Positive Rate (recall) vs. False Positive Rate ($FP / (FP + TN)$) at all classification thresholds. The Area Under the Curve (AUC) measures the probability that a randomly chosen positive example is ranked higher than a randomly chosen negative example. AUC = 0.5 indicates random performance; AUC = 1.0 indicates perfect ranking.

**AUC-PR (Precision-Recall).** For highly imbalanced datasets, AUC-PR is more informative than AUC-ROC because it focuses on the minority (positive) class. The PR curve plots precision vs. recall at all thresholds.

**Log Loss (Binary Cross-Entropy).** Measures the quality of probabilistic predictions:

$$\text{Log Loss} = -\frac{1}{n}\sum_{i=1}^{n}[y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)]$$

Log loss penalizes confident wrong predictions severely.

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, log_loss,
                             classification_report, confusion_matrix,
                             average_precision_score)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"AUC-PR: {average_precision_score(y_test, y_proba):.4f}")
print(f"Log Loss: {log_loss(y_test, y_proba):.4f}")
```

### 5.11.2 Cross-Validation

**K-Fold Cross-Validation.** The data is split into $K$ folds. The model is trained on $K-1$ folds and evaluated on the held-out fold. This is repeated $K$ times, and results are averaged. Common choices: $K = 5$ or $K = 10$.

**Stratified K-Fold.** Ensures each fold has approximately the same class distribution as the full dataset. Essential for imbalanced classification.

**Time Series Split.** For temporal data, future data must never leak into training. `TimeSeriesSplit` uses expanding windows: fold $k$ trains on data from time 1 to $t_k$ and evaluates on time $t_k + 1$ to $t_{k+1}$.

```python
from sklearn.model_selection import (KFold, StratifiedKFold,
                                     TimeSeriesSplit, cross_val_score)

# Stratified K-Fold for imbalanced classification
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
scores_ts = cross_val_score(model, X_ts, y_ts, cv=tscv, scoring='neg_mean_squared_error')
```

### 5.11.3 Learning Curves and the Bias-Variance Tradeoff

The **bias-variance decomposition** states that the expected prediction error for any model can be decomposed as:

$$\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

- **High bias (underfitting)**: The model is too simple. Both training and validation errors are high.
- **High variance (overfitting)**: The model is too complex. Training error is low, but validation error is high.

**Learning curves** plot training and validation performance as a function of training set size. They diagnose whether the problem is bias-dominated (both curves plateau at a poor level) or variance-dominated (a large gap between curves).

```python
from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
# Plot train_scores.mean(axis=1) and val_scores.mean(axis=1) vs train_sizes
```

---

## 5.12 Calibration

A classifier is **well-calibrated** if, among all instances it assigns probability $p$, approximately a fraction $p$ actually belong to the positive class. Calibration is critical when predicted probabilities are used for downstream decision-making (e.g., medical diagnosis, credit risk).

### 5.12.1 Reliability Diagrams

A reliability diagram (calibration curve) bins predicted probabilities and plots the mean predicted probability against the observed frequency. A perfectly calibrated model follows the diagonal.

### 5.12.2 Brier Score

The Brier score measures the mean squared error of probabilistic predictions:

$$\text{Brier} = \frac{1}{n}\sum_{i=1}^{n}(\hat{p}_i - y_i)^2$$

Lower is better. The Brier score can be decomposed into calibration, refinement, and uncertainty components.

### 5.12.3 Calibration Methods

**Platt Scaling** (Platt, 1999) fits a logistic regression on the model's outputs to produce calibrated probabilities. It works well when the calibration function is sigmoid-shaped.

**Isotonic Regression** fits a non-decreasing step function. It is more flexible than Platt scaling but requires more data to avoid overfitting.

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

# Calibrate using Platt scaling (sigmoid)
calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=5)
calibrated.fit(X_train, y_train)

y_proba_cal = calibrated.predict_proba(X_test)[:, 1]
print(f"Brier score (uncalibrated): {brier_score_loss(y_test, y_proba):.4f}")
print(f"Brier score (calibrated): {brier_score_loss(y_test, y_proba_cal):.4f}")

# Reliability diagram data
prob_true, prob_pred = calibration_curve(y_test, y_proba_cal, n_bins=10)
```

---

## 5.13 Handling Class Imbalance

Many real-world problems involve imbalanced classes: fraud detection (< 1% fraud), medical diagnosis (rare diseases), anomaly detection. Standard algorithms trained on imbalanced data tend to predict the majority class, achieving high accuracy but poor recall on the minority class.

### 5.13.1 Data-Level Methods

**SMOTE** (Synthetic Minority Over-sampling Technique) (Chawla et al., 2002) generates synthetic minority samples by interpolating between existing minority samples and their k-nearest neighbors:

$$\mathbf{x}_{\text{new}} = \mathbf{x}_i + \lambda (\mathbf{x}_j - \mathbf{x}_i), \quad \lambda \sim \text{Uniform}(0, 1)$$

where $\mathbf{x}_j$ is a randomly selected nearest neighbor of $\mathbf{x}_i$.

**ADASYN** (Adaptive Synthetic Sampling) focuses on generating samples near decision boundaries by oversampling more in regions with lower minority density (He et al., 2008).

```python
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline

# Always resample only the training data, never the test data
smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
pipeline = ImbPipeline([
    ('smote', smote),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])
pipeline.fit(X_train, y_train)
```

### 5.13.2 Algorithm-Level Methods

**Class Weights.** Most scikit-learn classifiers accept a `class_weight` parameter. Setting `class_weight='balanced'` automatically adjusts weights inversely proportional to class frequencies: $w_k = n / (K \cdot n_k)$. This modifies the loss function so that each class contributes equally to the total loss regardless of its frequency. For example, in logistic regression, the weighted loss becomes:

$$\mathcal{L} = -\sum_{i=1}^{n} w_{y_i} \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right]$$

**Threshold Tuning.** The default 0.5 classification threshold is optimal only when classes are balanced and misclassification costs are symmetric. For imbalanced problems, threshold tuning is one of the most effective and underutilized strategies. The procedure is:

1. Train the model and obtain predicted probabilities on a validation set.
2. Sweep thresholds from 0 to 1 in small increments.
3. Compute the target metric (e.g., F1, or recall at a minimum precision) at each threshold.
4. Select the threshold that maximizes the target metric.

This approach is especially effective because it does not modify the training process or the data distribution --- it simply adjusts the decision boundary to match the business requirements.

```python
from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np

precision, recall, thresholds = precision_recall_curve(y_val, y_proba_val)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_threshold = thresholds[np.argmax(f1_scores)]
y_pred_tuned = (y_proba_test >= optimal_threshold).astype(int)
print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"F1 with default threshold: {f1_score(y_test, y_proba_test >= 0.5):.3f}")
print(f"F1 with tuned threshold: {f1_score(y_test, y_pred_tuned):.3f}")
```

**Cost-Sensitive Learning.** Assign different misclassification costs to different classes, modifying the loss function to penalize minority class errors more heavily. This generalizes class weighting by allowing asymmetric costs: a false negative (missing a fraudulent transaction) may be far more costly than a false positive (flagging a legitimate transaction).

---

## 5.14 Hyperparameter Optimization

Model performance depends critically on hyperparameters --- settings that are not learned during training but must be specified by the practitioner.

### 5.14.1 Grid Search

Grid search exhaustively evaluates all combinations of hyperparameter values from a predefined grid. It is reliable but computationally expensive: for $k$ hyperparameters with $m$ values each, it requires $m^k$ evaluations.

### 5.14.2 Random Search

Random search (Bergstra & Bengio, 2012) samples hyperparameter values randomly from specified distributions. It is more efficient than grid search when only a few hyperparameters significantly affect performance --- random search is more likely to find good values for the important hyperparameters.

### 5.14.3 Bayesian Optimization

Bayesian optimization builds a probabilistic surrogate model of the objective function and uses an acquisition function to select the next point to evaluate. The Tree-structured Parzen Estimator (TPE), implemented in Optuna (Akiba et al., 2019), models $P(\text{hyperparameters} | \text{score} < \text{threshold})$ and $P(\text{hyperparameters} | \text{score} \geq \text{threshold})$ separately, selecting hyperparameters that maximize the ratio.

```python
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3,
                                             log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20)
    }
    model = GradientBoostingClassifier(**params, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5,
                             scoring='roc_auc')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)
print(f"Best AUC-ROC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

### 5.14.4 Hyperband

Hyperband (Li et al., 2017) addresses the resource allocation problem: rather than training all configurations to completion, it uses successive halving to quickly eliminate poor configurations. Configurations are trained with increasing budgets, and the bottom half is discarded at each round. Hyperband runs multiple rounds of successive halving with different initial budgets to balance breadth and depth.

Optuna integrates Hyperband via the `MedianPruner` or `HyperbandPruner`:

```python
study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.HyperbandPruner(
        min_resource=1, max_resource=100, reduction_factor=3
    )
)
```

---

## 5.15 AutoML

AutoML systems automate the end-to-end machine learning pipeline: feature preprocessing, algorithm selection, hyperparameter tuning, and model ensembling.

### 5.15.1 H2O AutoML

H2O AutoML trains and cross-validates a portfolio of models (GLMs, Random Forests, GBMs, Deep Learning, XGBoost, Stacked Ensembles) and ranks them by a user-specified metric:

```python
import h2o
from h2o.automl import H2OAutoML

h2o.init()
df = h2o.import_file("data.csv")
train, test = df.split_frame(ratios=[0.8])

aml = H2OAutoML(max_runtime_secs=3600, seed=42, sort_metric='AUC')
aml.train(x=features, y=target, training_frame=train)
lb = aml.leaderboard
print(lb.head(10))
```

### 5.15.2 AutoGluon

AutoGluon (Erickson et al., 2020) from Amazon emphasizes ease of use and multi-layer stacking. A single line of code trains a high-quality model:

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label='target', eval_metric='roc_auc')
predictor.fit(train_data, time_limit=3600, presets='best_quality')
results = predictor.evaluate(test_data)
```

### 5.15.3 FLAML

FLAML (Wang et al., 2021) from Microsoft focuses on cost-effective hyperparameter optimization using a novel cost-aware search strategy. It achieves competitive performance with significantly less compute than other AutoML systems.

### 5.15.4 When to Use AutoML vs. Manual Tuning

AutoML is appropriate when:
- You need a strong baseline quickly.
- The problem is standard (tabular classification/regression).
- Compute resources are available.

Manual tuning is preferred when:
- You need interpretability and control.
- Domain knowledge suggests specific architectures or features.
- The problem involves custom loss functions, constraints, or deployment requirements.
- You need to understand why the model works.

The pragmatic approach: use AutoML for a strong baseline, then manually improve upon it with domain-specific feature engineering and targeted hyperparameter tuning.

---

## Exercises

1. **Mathematical Foundations**: Derive the gradient of the Ridge regression loss function and show that the closed-form solution reduces to OLS when $\alpha = 0$.

2. **Implementation**: Implement gradient descent for logistic regression from scratch (using only NumPy) on the breast cancer dataset. Compare convergence behavior for different learning rates. Verify your results against `sklearn.linear_model.LogisticRegression`.

3. **SVM Exploration**: Using the `make_circles` dataset, demonstrate that a linear SVM fails while an RBF kernel SVM succeeds. Visualize the decision boundaries. Explain why the kernel trick is necessary.

4. **Ensemble Comparison**: On a dataset of your choice, compare the performance of a single Decision Tree, Random Forest (100 trees), AdaBoost, and Gradient Boosting. Plot learning curves for each. Which method benefits most from additional data?

5. **Gradient Boosting Frameworks**: Using the same dataset, compare XGBoost, LightGBM, and CatBoost in terms of (a) accuracy, (b) training time, and (c) ease of handling categorical features. Use Optuna to tune each.

6. **Clustering**: Generate a dataset with `make_blobs` (5 clusters, different standard deviations). Apply K-Means, DBSCAN, and a GMM. When does DBSCAN outperform K-Means? Visualize with PCA and UMAP.

7. **Feature Engineering Pipeline**: Build a complete scikit-learn `Pipeline` with `ColumnTransformer` that handles numerical features (imputation + scaling), categorical features (target encoding), and applies feature selection. Use this pipeline inside a `GridSearchCV`.

8. **Calibration**: Train a Random Forest on the breast cancer dataset. Plot the reliability diagram before and after Platt scaling. Compute and compare the Brier scores. When would you prefer isotonic regression over Platt scaling?

9. **Class Imbalance**: On a highly imbalanced dataset (e.g., credit card fraud), compare (a) no treatment, (b) SMOTE, (c) class weighting, and (d) threshold tuning. Which approach yields the best F1 on the minority class? Which yields the best AUC-PR?

10. **Hyperparameter Optimization**: Compare grid search, random search, and Bayesian optimization (Optuna) for tuning a LightGBM model. Plot the best score as a function of number of trials. At what trial count does Bayesian optimization overtake random search?

---

## References

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2623--2631.

Arthur, D., & Vassilvitskii, S. (2007). k-means++: The Advantages of Careful Seeding. *Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete Algorithms*, 1027--1035.

Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. *Journal of Machine Learning Research*, 13, 281--305.

Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

Breiman, L. (1996). Bagging Predictors. *Machine Learning*, 24(2), 123--140.

Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5--32.

Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). *Classification and Regression Trees*. CRC Press.

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321--357.

Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785--794.

Cortes, C., & Vapnik, V. (1995). Support-Vector Networks. *Machine Learning*, 20(3), 273--297.

Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum Likelihood from Incomplete Data via the EM Algorithm. *Journal of the Royal Statistical Society: Series B*, 39(1), 1--38.

Erickson, N., Mueller, J., Shirkov, A., et al. (2020). AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data. *arXiv preprint arXiv:2003.06505*.

Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise. *Proceedings of the 2nd International Conference on Knowledge Discovery and Data Mining*, 226--231.

Frenay, B., & Verleysen, M. (2014). Classification in the Presence of Label Noise: A Survey. *IEEE Transactions on Neural Networks and Learning Systems*, 25(5), 845--869.

Freund, Y., & Schapire, R. E. (1997). A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting. *Journal of Computer and System Sciences*, 55(1), 119--139.

Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics*, 29(5), 1189--1232.

Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.

He, H., Bai, Y., Garcia, E. A., & Li, S. (2008). ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning. *Proceedings of the IEEE International Joint Conference on Neural Networks*, 1322--1328.

Hoerl, A. E., & Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. *Technometrics*, 12(1), 55--67.

Hotelling, H. (1933). Analysis of a Complex of Statistical Variables into Principal Components. *Journal of Educational Psychology*, 24(6), 417--441.

Ke, G., Meng, Q., Finley, T., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems*, 30, 3146--3154.

Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2017). Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization. *Journal of Machine Learning Research*, 18(185), 1--52.

Lloyd, S. (1982). Least Squares Quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129--137.

McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv preprint arXiv:1802.03426*.

Pearson, K. (1901). On Lines and Planes of Closest Fit to Systems of Points in Space. *The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science*, 2(11), 559--572.

Platt, J. C. (1999). Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods. *Advances in Large Margin Classifiers*, 10(3), 61--74.

Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased Boosting with Categorical Features. *Advances in Neural Information Processing Systems*, 31, 6638--6648.

Quionero-Candela, J., Sugiyama, M., Schwaighofer, A., & Lawrence, N. D. (2009). *Dataset Shift in Machine Learning*. MIT Press.

Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann.

Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267--288.

van Buuren, S., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate Imputation by Chained Equations in R. *Journal of Statistical Software*, 45(3), 1--67.

van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. *Journal of Machine Learning Research*, 9, 2579--2605.

Wang, C., Wu, Q., Weimer, M., & Zhu, E. (2021). FLAML: A Fast and Lightweight AutoML Library. *Proceedings of Machine Learning and Systems*, 3, 434--447.

Wolpert, D. H. (1992). Stacked Generalization. *Neural Networks*, 5(2), 241--259.

Wolpert, D. H. (1996). The Lack of A Priori Distinctions Between Learning Algorithms. *Neural Computation*, 8(7), 1341--1390.

Zheng, A., & Casari, A. (2018). *Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists*. O'Reilly Media.

Zou, H., & Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301--320.
