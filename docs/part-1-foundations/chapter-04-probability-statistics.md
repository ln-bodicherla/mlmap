# Chapter 4: Probability and Statistics

> *"Probability theory is nothing but common sense reduced to calculation."*
> — Pierre-Simon Laplace

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. State the axioms of probability and reason rigorously about sample spaces, events, and sigma-algebras.
2. Apply Bayes' theorem to real ML problems including spam filtering, medical diagnosis, and model updating.
3. Identify and work with key probability distributions (Normal, Bernoulli, Binomial, Categorical, Poisson, Exponential, Beta, Dirichlet) and recognize where each appears in ML.
4. Compute expectations, variances, covariances, and higher moments, and use them to characterize distributions.
5. Derive maximum likelihood estimators (MLE) for Gaussian and Bernoulli distributions and connect MLE to common loss functions.
6. Derive maximum a posteriori (MAP) estimators and demonstrate their equivalence to regularized optimization.
7. Use information-theoretic concepts (entropy, cross-entropy, KL divergence) to understand and derive loss functions.
8. Perform hypothesis tests (t-test, KS test, chi-squared, PSI) and apply them to ML model evaluation and drift detection.
9. Describe Bayesian inference including priors, posteriors, conjugate priors, and MCMC sampling methods.
10. Explain probabilistic graphical models (Bayesian networks, HMMs, CRFs) and their role in structured prediction.

---

## 4.1 Probability Axioms and Foundations

### 4.1.1 Sample Spaces and Events

A **probability space** consists of three elements $(\Omega, \mathcal{F}, P)$:

- **Sample space** $\Omega$: the set of all possible outcomes of a random experiment. For a coin flip: $\Omega = \{H, T\}$. For a continuous measurement: $\Omega = \mathbb{R}$. For an image: $\Omega = [0, 255]^{H \times W \times 3}$.

- **Event space** $\mathcal{F}$ (sigma-algebra): a collection of subsets of $\Omega$ to which we can assign probabilities. It must satisfy: (1) $\Omega \in \mathcal{F}$, (2) if $A \in \mathcal{F}$ then $A^c \in \mathcal{F}$ (closure under complements), (3) if $A_1, A_2, \ldots \in \mathcal{F}$ then $\bigcup_i A_i \in \mathcal{F}$ (closure under countable unions).

- **Probability measure** $P: \mathcal{F} \to [0, 1]$: a function assigning probabilities to events.

### 4.1.2 Kolmogorov's Axioms

1. **Non-negativity:** $P(A) \geq 0$ for all $A \in \mathcal{F}$.
2. **Normalization:** $P(\Omega) = 1$.
3. **Countable additivity:** For mutually exclusive events $A_1, A_2, \ldots$: $P\left(\bigcup_i A_i\right) = \sum_i P(A_i)$.

From these three axioms, everything else follows:
- $P(\emptyset) = 0$
- $P(A^c) = 1 - P(A)$
- $P(A \cup B) = P(A) + P(B) - P(A \cap B)$
- If $A \subseteq B$, then $P(A) \leq P(B)$

### 4.1.3 Random Variables

A **random variable** $X$ is a function from the sample space to the real numbers: $X: \Omega \to \mathbb{R}$. It maps outcomes to numbers, allowing us to do arithmetic with probabilities.

A **discrete** random variable takes countably many values. Its distribution is characterized by the **probability mass function (PMF)**: $p(x) = P(X = x)$.

A **continuous** random variable takes values in an interval. Its distribution is characterized by the **probability density function (PDF)** $f(x)$ where $P(a \leq X \leq b) = \int_a^b f(x) dx$.

The **cumulative distribution function (CDF)** $F(x) = P(X \leq x)$ exists for both discrete and continuous random variables.

---

## 4.2 Conditional Probability and Bayes' Theorem

### 4.2.1 Conditional Probability

The probability of $A$ given that $B$ has occurred:

$$P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

**Independence:** $A$ and $B$ are independent if $P(A|B) = P(A)$, equivalently $P(A \cap B) = P(A)P(B)$.

**Conditional independence:** $A$ and $B$ are conditionally independent given $C$ if $P(A \cap B | C) = P(A|C)P(B|C)$. This is weaker than independence and is the foundation of graphical models.

### 4.2.2 Bayes' Theorem

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} = \frac{P(B|A) \cdot P(A)}{\sum_{a} P(B|A=a) \cdot P(A=a)}$$

In ML terminology:

$$\underbrace{P(\theta | \text{data})}_{\text{posterior}} = \frac{\underbrace{P(\text{data} | \theta)}_{\text{likelihood}} \cdot \underbrace{P(\theta)}_{\text{prior}}}{\underbrace{P(\text{data})}_{\text{evidence}}}$$

**Example: Spam Classification**

Let $S$ = email is spam, $W$ = email contains "congratulations".

Given: $P(S) = 0.3$, $P(W|S) = 0.8$, $P(W|\neg S) = 0.1$.

$$P(S|W) = \frac{P(W|S) \cdot P(S)}{P(W|S) \cdot P(S) + P(W|\neg S) \cdot P(\neg S)} = \frac{0.8 \times 0.3}{0.8 \times 0.3 + 0.1 \times 0.7} = \frac{0.24}{0.31} \approx 0.774$$

An email containing "congratulations" has a 77.4% chance of being spam.

```python
def bayes_theorem(prior, likelihood, likelihood_complement):
    """
    Compute posterior probability using Bayes' theorem.

    P(A|B) = P(B|A) * P(A) / [P(B|A)*P(A) + P(B|~A)*P(~A)]
    """
    evidence = likelihood * prior + likelihood_complement * (1 - prior)
    posterior = (likelihood * prior) / evidence
    return posterior

# Spam example
posterior = bayes_theorem(
    prior=0.3,               # P(spam)
    likelihood=0.8,           # P(word|spam)
    likelihood_complement=0.1 # P(word|not spam)
)
print(f"P(spam | word) = {posterior:.3f}")  # 0.774
```

**Example: Medical Testing and Base Rate Fallacy**

A medical test has 99% sensitivity (true positive rate) and 99% specificity (true negative rate). The disease prevalence is 0.1%. What is the probability that a positive test result actually means the patient has the disease?

$$P(\text{disease}|\text{positive}) = \frac{0.99 \times 0.001}{0.99 \times 0.001 + 0.01 \times 0.999} = \frac{0.00099}{0.01098} \approx 0.09$$

Only 9%! Despite the test's high accuracy, the low base rate means most positive results are false positives. This is the **base rate fallacy** — and it appears in ML whenever class imbalance is present.

---

## 4.3 Common Probability Distributions

### 4.3.1 Bernoulli Distribution

A single binary outcome (coin flip, spam/not-spam, click/no-click).

$$P(X = x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}$$

$$\mathbb{E}[X] = p, \quad \text{Var}(X) = p(1-p)$$

**In ML:** Binary classification output (single sigmoid neuron), dropout (each neuron has a Bernoulli mask).

### 4.3.2 Binomial Distribution

The number of successes in $n$ independent Bernoulli trials.

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n$$

$$\mathbb{E}[X] = np, \quad \text{Var}(X) = np(1-p)$$

**In ML:** Number of correct predictions in a batch, A/B testing (number of conversions).

### 4.3.3 Categorical Distribution

Generalization of Bernoulli to $K$ classes. A single draw from $K$ categories with probabilities $p_1, \ldots, p_K$ where $\sum_k p_k = 1$.

$$P(X = k) = p_k$$

Represented as a one-hot vector $\mathbf{x} \in \{0,1\}^K$ with $\sum_k x_k = 1$:

$$P(\mathbf{x}) = \prod_{k=1}^{K} p_k^{x_k}$$

**In ML:** Multi-class classification (softmax output), language model next-token prediction, multinomial naive Bayes.

### 4.3.4 Normal (Gaussian) Distribution

The most important continuous distribution, by the Central Limit Theorem.

$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

$$\mathbb{E}[X] = \mu, \quad \text{Var}(X) = \sigma^2$$

The **multivariate Gaussian** in $\mathbb{R}^d$:

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

**In ML:** Weight initialization (Gaussian random), noise in GANs and diffusion models, Gaussian Mixture Models, the latent space of VAEs, GP regression.

### 4.3.5 Poisson Distribution

Models the number of events in a fixed time period, given a constant average rate $\lambda$.

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$

$$\mathbb{E}[X] = \lambda, \quad \text{Var}(X) = \lambda$$

**In ML:** Count data modeling (number of website visits, word occurrences), Poisson regression for count outcomes.

### 4.3.6 Exponential Distribution

Models the time between events in a Poisson process.

$$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$

$$\mathbb{E}[X] = 1/\lambda, \quad \text{Var}(X) = 1/\lambda^2$$

**In ML:** Modeling inter-arrival times, survival analysis, time-to-event prediction.

### 4.3.7 Beta Distribution

A distribution over probabilities $p \in [0, 1]$, parameterized by shape parameters $\alpha, \beta > 0$.

$$f(p) = \frac{p^{\alpha-1}(1-p)^{\beta-1}}{B(\alpha, \beta)}$$

where $B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$ is the Beta function.

$$\mathbb{E}[p] = \frac{\alpha}{\alpha+\beta}, \quad \text{Var}(p) = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$$

**In ML:** The conjugate prior for the Bernoulli/Binomial likelihood. Bayesian A/B testing. Thompson sampling in bandits. The Beta($\alpha$=1, $\beta$=1) distribution is uniform on [0, 1].

### 4.3.8 Dirichlet Distribution

The multivariate generalization of the Beta distribution. A distribution over probability simplices (vectors that sum to 1).

$$f(\mathbf{p}) = \frac{1}{B(\boldsymbol{\alpha})} \prod_{k=1}^{K} p_k^{\alpha_k - 1}, \quad \sum_k p_k = 1$$

$$\mathbb{E}[p_k] = \frac{\alpha_k}{\sum_j \alpha_j}$$

**In ML:** Prior for topic distributions in LDA (Latent Dirichlet Allocation), conjugate prior for the Categorical/Multinomial likelihood.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Visualize key distributions
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Bernoulli
x = [0, 1]
for p in [0.3, 0.5, 0.7]:
    axes[0, 0].bar([v + 0.1*(p-0.5) for v in x], [1-p, p], width=0.1,
                   label=f'p={p}')
axes[0, 0].set_title('Bernoulli')
axes[0, 0].legend()

# Binomial
x = np.arange(0, 21)
for n, p in [(20, 0.3), (20, 0.5), (20, 0.7)]:
    axes[0, 1].plot(x, stats.binom.pmf(x, n, p), 'o-', label=f'n={n},p={p}')
axes[0, 1].set_title('Binomial')
axes[0, 1].legend()

# Poisson
x = np.arange(0, 20)
for lam in [1, 4, 10]:
    axes[0, 2].plot(x, stats.poisson.pmf(x, lam), 'o-', label=f'λ={lam}')
axes[0, 2].set_title('Poisson')
axes[0, 2].legend()

# Normal
x = np.linspace(-6, 6, 200)
for mu, sigma in [(0, 1), (0, 2), (2, 0.5)]:
    axes[0, 3].plot(x, stats.norm.pdf(x, mu, sigma),
                    label=f'μ={mu},σ={sigma}')
axes[0, 3].set_title('Normal')
axes[0, 3].legend()

# Exponential
x = np.linspace(0, 5, 200)
for lam in [0.5, 1, 2]:
    axes[1, 0].plot(x, stats.expon.pdf(x, scale=1/lam), label=f'λ={lam}')
axes[1, 0].set_title('Exponential')
axes[1, 0].legend()

# Beta
x = np.linspace(0.001, 0.999, 200)
for a, b in [(0.5, 0.5), (2, 2), (2, 5), (5, 1)]:
    axes[1, 1].plot(x, stats.beta.pdf(x, a, b), label=f'α={a},β={b}')
axes[1, 1].set_title('Beta')
axes[1, 1].legend()

# Dirichlet (visualize as ternary — here we show sampled means)
# For K=3, sample from different concentration parameters
from scipy.stats import dirichlet
alphas = [[1, 1, 1], [10, 10, 10], [0.1, 0.1, 0.1]]
for alpha in alphas:
    samples = dirichlet.rvs(alpha, size=500)
    axes[1, 2].scatter(samples[:, 0], samples[:, 1], s=2,
                       label=f'α={alpha}', alpha=0.5)
axes[1, 2].set_title('Dirichlet (first 2 components)')
axes[1, 2].legend()

plt.tight_layout()
plt.show()
```

---

## 4.4 Expectation, Variance, Covariance, and Moments

### 4.4.1 Expectation

The **expected value** (mean) of a random variable is its probability-weighted average:

Discrete: $\mathbb{E}[X] = \sum_x x \cdot P(X = x)$

Continuous: $\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot f(x) \, dx$

**Key properties:**
- **Linearity:** $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$ (always, even if $X$, $Y$ are dependent)
- **Law of the Unconscious Statistician (LOTUS):** $\mathbb{E}[g(X)] = \sum_x g(x) P(X=x)$

### 4.4.2 Variance and Standard Deviation

$$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

$$\text{Var}(aX + b) = a^2 \text{Var}(X)$$

Standard deviation: $\sigma = \sqrt{\text{Var}(X)}$

### 4.4.3 Covariance and Correlation

**Covariance** measures the linear relationship between two random variables:

$$\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$$

If $X$ and $Y$ are independent, then $\text{Cov}(X,Y) = 0$. The converse is not true in general — zero covariance does not imply independence (it only rules out linear dependence).

**Correlation** normalizes covariance to $[-1, 1]$:

$$\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

The **covariance matrix** $\boldsymbol{\Sigma}$ of a random vector $\mathbf{X} \in \mathbb{R}^d$ has entries $\Sigma_{ij} = \text{Cov}(X_i, X_j)$. It is always symmetric and positive semi-definite. This matrix is central to PCA, Gaussian distributions, and Mahalanobis distance.

### 4.4.4 Moments

The $k$-th **moment** of $X$ is $\mathbb{E}[X^k]$. The $k$-th **central moment** is $\mathbb{E}[(X - \mu)^k]$.

- 1st moment: mean
- 2nd central moment: variance
- 3rd standardized central moment: **skewness** (asymmetry)
- 4th standardized central moment: **kurtosis** (tail heaviness)

The **moment generating function** $M_X(t) = \mathbb{E}[e^{tX}]$ encodes all moments: $\mathbb{E}[X^k] = M_X^{(k)}(0)$. Two distributions with the same MGF are identical (when it exists in a neighborhood of 0).

---

## 4.5 Maximum Likelihood Estimation (MLE)

### 4.5.1 The Principle

Given observed data $\mathcal{D} = \{x_1, x_2, \ldots, x_n\}$ assumed to be drawn independently from a distribution $p(x|\theta)$, the **maximum likelihood estimator** is the parameter value that maximizes the probability of the observed data:

$$\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} \prod_{i=1}^{n} p(x_i | \theta) = \arg\max_{\theta} \sum_{i=1}^{n} \log p(x_i | \theta)$$

The logarithm converts the product to a sum, which is more numerically stable and analytically convenient. Maximizing the log-likelihood is equivalent to minimizing the **negative log-likelihood (NLL)**.

### 4.5.2 MLE for the Gaussian Distribution

Given $\{x_1, \ldots, x_n\} \sim \mathcal{N}(\mu, \sigma^2)$:

$$\log p(\mathcal{D}|\mu, \sigma^2) = \sum_{i=1}^{n} \log \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)$$

$$= -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2$$

**Maximizing with respect to $\mu$:**

$$\frac{\partial}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i - \mu) = 0$$

$$\hat{\mu}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^{n} x_i = \bar{x}$$

The sample mean.

**Maximizing with respect to $\sigma^2$:**

$$\frac{\partial}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^{n}(x_i - \mu)^2 = 0$$

$$\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

Note: this is the **biased** estimate (divides by $n$, not $n-1$). The unbiased estimate uses $n-1$ (Bessel's correction).

### 4.5.3 MLE for the Bernoulli Distribution

Given $\{x_1, \ldots, x_n\} \sim \text{Bernoulli}(p)$:

$$\log p(\mathcal{D}|p) = \sum_{i=1}^{n} \left[x_i \log p + (1-x_i)\log(1-p)\right]$$

$$\frac{\partial}{\partial p} = \sum_{i=1}^{n}\left[\frac{x_i}{p} - \frac{1-x_i}{1-p}\right] = 0$$

$$\hat{p}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^{n} x_i = \bar{x}$$

The sample proportion.

### 4.5.4 MLE and Loss Functions: The Deep Connection

**MSE loss is the negative log-likelihood of a Gaussian:**

If we model $y = f_\theta(x) + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma^2)$, then $p(y|x,\theta) = \mathcal{N}(f_\theta(x), \sigma^2)$, and minimizing the NLL is equivalent to minimizing:

$$\text{NLL} \propto \sum_{i=1}^{n} (y_i - f_\theta(x_i))^2 = n \cdot \text{MSE}$$

**Cross-entropy loss is the negative log-likelihood of a Bernoulli (or Categorical):**

For binary classification with $y \in \{0, 1\}$ and predicted probability $\hat{p} = \sigma(z)$:

$$\text{NLL} = -\sum_{i=1}^{n}\left[y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right] = n \cdot \text{BCE}$$

This is binary cross-entropy (BCE). The generalization to $K$ classes gives categorical cross-entropy.

**This is a profound insight:** when we choose MSE for regression or cross-entropy for classification, we are not making arbitrary choices. We are performing maximum likelihood estimation under specific distributional assumptions (Bishop, 2006).

```python
import torch
import torch.nn as nn

# MLE connection: these are equivalent
# 1. Minimize MSE loss
mse_loss = nn.MSELoss()

# 2. Minimize negative Gaussian log-likelihood
gaussian_nll = nn.GaussianNLLLoss()

# For classification:
# 1. Minimize cross-entropy loss
ce_loss = nn.CrossEntropyLoss()

# 2. Minimize negative categorical log-likelihood
# (CrossEntropyLoss IS the NLL of Categorical with softmax)
```

---

## 4.6 MAP Estimation and Regularization

### 4.6.1 Maximum A Posteriori Estimation

MAP estimation adds a prior $p(\theta)$ to MLE:

$$\hat{\theta}_{\text{MAP}} = \arg\max_{\theta} p(\theta | \mathcal{D}) = \arg\max_{\theta} \left[\log p(\mathcal{D} | \theta) + \log p(\theta)\right]$$

The prior acts as a regularizer, encoding our beliefs about plausible parameter values.

### 4.6.2 Gaussian Prior = L2 Regularization (Ridge)

If $\theta \sim \mathcal{N}(0, \tau^2 \mathbf{I})$, then:

$$\log p(\theta) = -\frac{1}{2\tau^2}\|\theta\|_2^2 + \text{const}$$

So the MAP objective becomes:

$$\hat{\theta}_{\text{MAP}} = \arg\min_{\theta} \left[\underbrace{-\log p(\mathcal{D}|\theta)}_{\text{loss}} + \underbrace{\frac{1}{2\tau^2}\|\theta\|_2^2}_{\text{L2 penalty}}\right]$$

Setting $\lambda = \frac{1}{2\tau^2}$, this is exactly L2 regularization. A small $\tau$ (tight prior) corresponds to strong regularization (large $\lambda$); a large $\tau$ (vague prior) gives weak regularization.

### 4.6.3 Laplace Prior = L1 Regularization (Lasso)

If $\theta_j \sim \text{Laplace}(0, b)$ with PDF $\frac{1}{2b}\exp(-|x|/b)$, then:

$$\log p(\theta) = -\frac{1}{b}\sum_j |\theta_j| + \text{const}$$

This gives L1 regularization, which promotes sparsity because the Laplace prior has a sharp peak at zero.

```python
# MAP estimation is regularized MLE
# The prior determines the type of regularization

# L2 regularization (Gaussian prior) in PyTorch
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)
# weight_decay adds lambda * ||w||^2 to the loss

# L1 regularization (Laplace prior) — manual implementation
l1_lambda = 0.001
loss = criterion(predictions, targets)
l1_norm = sum(p.abs().sum() for p in model.parameters())
loss_with_l1 = loss + l1_lambda * l1_norm
loss_with_l1.backward()
```

---

## 4.7 Information Theory

Information theory, developed by Claude Shannon (1948), provides a mathematical framework for quantifying information, uncertainty, and the difference between probability distributions. These concepts directly underpin the loss functions used in modern ML.

### 4.7.1 Entropy

The **entropy** of a discrete random variable $X$ with PMF $p$ is:

$$H(X) = -\sum_x p(x) \log p(x)$$

Entropy measures the average amount of "surprise" or "information" in each observation. A deterministic variable has $H = 0$ (no surprise); a uniform distribution over $K$ outcomes has maximum entropy $H = \log K$.

Properties:
- $H(X) \geq 0$
- $H(X) \leq \log |\mathcal{X}|$ (maximized by uniform distribution)
- $H(X, Y) \leq H(X) + H(Y)$ (equality iff $X$, $Y$ are independent)

For continuous distributions, the **differential entropy** is:

$$h(X) = -\int f(x) \log f(x) \, dx$$

The Gaussian has the maximum differential entropy among all distributions with the same variance:

$$h(\mathcal{N}(\mu, \sigma^2)) = \frac{1}{2}\log(2\pi e \sigma^2)$$

### 4.7.2 Cross-Entropy

The **cross-entropy** between the true distribution $p$ and an approximate distribution $q$ is:

$$H(p, q) = -\sum_x p(x) \log q(x) = H(p) + D_{\text{KL}}(p \| q)$$

Since $H(p)$ is constant with respect to $q$, minimizing cross-entropy is equivalent to minimizing KL divergence.

**This is why we use cross-entropy loss in classification:** the true distribution $p$ is the one-hot label vector, and $q$ is the model's predicted probability distribution. Minimizing cross-entropy makes the model's predictions as close as possible (in KL sense) to the true labels.

$$\mathcal{L}_{\text{CE}} = -\sum_{k=1}^{K} y_k \log \hat{p}_k = -\log \hat{p}_c$$

where $c$ is the correct class. This reduces to the negative log-probability of the correct class.

### 4.7.3 KL Divergence

The **Kullback-Leibler divergence** from $q$ to $p$ is:

$$D_{\text{KL}}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \mathbb{E}_{p}\left[\log \frac{p(x)}{q(x)}\right]$$

Properties:
- $D_{\text{KL}}(p \| q) \geq 0$ (Gibbs' inequality)
- $D_{\text{KL}}(p \| q) = 0$ if and only if $p = q$
- **Not symmetric:** $D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p)$ in general
- Not a metric (violates symmetry and triangle inequality)

The asymmetry has practical consequences:
- **$D_{\text{KL}}(p \| q)$ (forward KL)**: Penalizes $q$ for placing low probability where $p$ is high. Forces $q$ to be "mean-seeking" — cover all modes of $p$. Used in variational inference (ELBO).
- **$D_{\text{KL}}(q \| p)$ (reverse KL)**: Penalizes $q$ for placing high probability where $p$ is low. Forces $q$ to be "mode-seeking" — lock onto one mode. Used in reinforcement learning (KL penalty in PPO).

**KL divergence for Gaussians** (used in VAE loss):

$$D_{\text{KL}}(\mathcal{N}(\mu_1, \sigma_1^2) \| \mathcal{N}(\mu_2, \sigma_2^2)) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

For the special case $\mathcal{N}(\mu, \sigma^2)$ vs. $\mathcal{N}(0, 1)$ (VAE prior):

$$D_{\text{KL}}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1)) = -\frac{1}{2}\left(1 + \log\sigma^2 - \mu^2 - \sigma^2\right)$$

```python
import torch
import torch.nn.functional as F

# Cross-entropy loss
logits = torch.randn(32, 10)       # Batch of 32, 10 classes
targets = torch.randint(0, 10, (32,))
loss = F.cross_entropy(logits, targets)  # Combines log_softmax + NLL

# KL divergence between two distributions
p = F.softmax(torch.randn(10), dim=0)
q = F.softmax(torch.randn(10), dim=0)
kl_div = F.kl_div(q.log(), p, reduction='sum')  # KL(p || q)

# Manual computation
kl_manual = (p * (p.log() - q.log())).sum()
print(f"KL divergence: {kl_div:.4f} (manual: {kl_manual:.4f})")

# VAE KL loss (Gaussian vs. standard normal)
def vae_kl_loss(mu, log_var):
    """KL(N(mu, sigma^2) || N(0, 1))"""
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
```

---

## 4.8 Hypothesis Testing

Hypothesis testing provides a framework for making decisions under uncertainty. In ML, it is used to compare models, detect data drift, validate assumptions, and assess statistical significance of results.

### 4.8.1 The Framework

1. State the **null hypothesis** $H_0$ (the "default" or "nothing interesting" hypothesis) and the **alternative hypothesis** $H_1$.
2. Choose a **test statistic** that measures how far the data is from what $H_0$ predicts.
3. Compute the **p-value**: the probability of observing a test statistic at least as extreme as the one computed, assuming $H_0$ is true.
4. **Reject** $H_0$ if the p-value is below a significance level $\alpha$ (typically 0.05).

### 4.8.2 Student's t-Test

Tests whether two groups have different means.

**Two-sample t-test** (are the means of groups A and B different?):

$$t = \frac{\bar{X}_A - \bar{X}_B}{\sqrt{\frac{s_A^2}{n_A} + \frac{s_B^2}{n_B}}}$$

```python
from scipy import stats

# Compare accuracy of two models
model_a_scores = [0.85, 0.87, 0.83, 0.86, 0.84, 0.88, 0.85, 0.87, 0.86, 0.84]
model_b_scores = [0.88, 0.90, 0.87, 0.89, 0.91, 0.88, 0.90, 0.89, 0.87, 0.88]

t_stat, p_value = stats.ttest_ind(model_a_scores, model_b_scores)
print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Significant difference between models (reject H0)")
else:
    print("No significant difference (fail to reject H0)")

# Paired t-test (same data, different models)
t_stat, p_value = stats.ttest_rel(model_a_scores, model_b_scores)
```

### 4.8.3 Kolmogorov-Smirnov (KS) Test

Tests whether two samples come from the same distribution (non-parametric — no assumptions about distribution shape). The test statistic is the maximum difference between the empirical CDFs:

$$D = \sup_x |F_A(x) - F_B(x)|$$

```python
# KS test for data drift detection
train_feature = np.random.normal(0, 1, 10000)     # Training distribution
prod_feature = np.random.normal(0.1, 1.1, 10000)  # Production distribution

ks_stat, p_value = stats.ks_2samp(train_feature, prod_feature)
print(f"KS statistic: {ks_stat:.4f}, p-value: {p_value:.6f}")
# Low p-value → distributions are different → drift detected
```

### 4.8.4 Chi-Squared Test

Tests the association between categorical variables or the goodness of fit to an expected distribution:

$$\chi^2 = \sum_i \frac{(O_i - E_i)^2}{E_i}$$

where $O_i$ are observed counts and $E_i$ are expected counts.

```python
# Chi-squared test: is the class distribution the same in train and prod?
train_counts = np.array([300, 400, 300])   # Classes 0, 1, 2
prod_counts = np.array([250, 500, 250])

chi2, p_value = stats.chisquare(prod_counts, f_exp=train_counts *
                                 prod_counts.sum() / train_counts.sum())
print(f"Chi-squared: {chi2:.2f}, p-value: {p_value:.4f}")
```

### 4.8.5 Population Stability Index (PSI) for Drift Detection

PSI is widely used in industry (especially finance) to detect distribution shift between training and serving data:

$$\text{PSI} = \sum_{i=1}^{B} (p_i^{\text{new}} - p_i^{\text{ref}}) \cdot \ln\frac{p_i^{\text{new}}}{p_i^{\text{ref}}}$$

where the continuous feature is binned into $B$ buckets, and $p_i$ is the proportion in bucket $i$.

Rules of thumb: PSI < 0.1 (no drift), 0.1-0.25 (moderate drift), > 0.25 (significant drift).

```python
def compute_psi(reference, current, n_bins=10):
    """Compute Population Stability Index."""
    # Create bins from reference distribution
    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    # Compute proportions in each bin
    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current, bins=breakpoints)[0]

    # Avoid zero proportions
    ref_pct = (ref_counts + 1) / (len(reference) + n_bins)
    cur_pct = (cur_counts + 1) / (len(current) + n_bins)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return psi

# Example: detect drift
reference = np.random.normal(0, 1, 10000)
no_drift = np.random.normal(0, 1, 10000)
drifted = np.random.normal(0.5, 1.5, 10000)

print(f"PSI (no drift): {compute_psi(reference, no_drift):.4f}")
print(f"PSI (drifted): {compute_psi(reference, drifted):.4f}")
```

---

## 4.9 Bayesian Inference

While MLE and MAP give point estimates, full Bayesian inference maintains the entire posterior distribution over parameters. This provides uncertainty quantification — we know not just "what" the model predicts, but "how confident" it is.

### 4.9.1 The Bayesian Framework

$$\underbrace{p(\theta|\mathcal{D})}_{\text{posterior}} = \frac{\underbrace{p(\mathcal{D}|\theta)}_{\text{likelihood}} \cdot \underbrace{p(\theta)}_{\text{prior}}}{\underbrace{p(\mathcal{D})}_{\text{evidence}}}$$

The **evidence** $p(\mathcal{D}) = \int p(\mathcal{D}|\theta) p(\theta) d\theta$ is typically intractable (requires integrating over all possible parameter values). This is the fundamental challenge of Bayesian inference.

Predictions integrate over the posterior:

$$p(x^* | \mathcal{D}) = \int p(x^* | \theta) p(\theta | \mathcal{D}) d\theta$$

### 4.9.2 Conjugate Priors

A prior is **conjugate** to a likelihood if the posterior has the same functional form as the prior. This allows analytical Bayesian updates.

**Beta-Bernoulli conjugacy:**

Prior: $p \sim \text{Beta}(\alpha, \beta)$

Data: $k$ successes in $n$ Bernoulli trials

Posterior: $p|\mathcal{D} \sim \text{Beta}(\alpha + k, \beta + n - k)$

The posterior simply adds the observed counts to the prior pseudo-counts.

```python
from scipy.stats import beta
import matplotlib.pyplot as plt

# Bayesian coin estimation
# Prior: Beta(2, 2) — mild belief that p ≈ 0.5
alpha_prior, beta_prior = 2, 2

# Observe: 7 heads out of 10 flips
n_heads, n_flips = 7, 10

# Posterior: Beta(2+7, 2+3) = Beta(9, 5)
alpha_post = alpha_prior + n_heads
beta_post = beta_prior + (n_flips - n_heads)

x = np.linspace(0, 1, 200)
plt.figure(figsize=(10, 6))
plt.plot(x, beta.pdf(x, alpha_prior, beta_prior), label='Prior: Beta(2,2)', ls='--')
plt.plot(x, beta.pdf(x, alpha_post, beta_post), label=f'Posterior: Beta({alpha_post},{beta_post})')
plt.axvline(n_heads/n_flips, color='red', ls=':', label=f'MLE: {n_heads/n_flips:.2f}')
plt.axvline(alpha_post/(alpha_post+beta_post), color='green', ls=':',
            label=f'Posterior mean: {alpha_post/(alpha_post+beta_post):.3f}')
plt.xlabel('p (probability of heads)')
plt.ylabel('Density')
plt.title('Bayesian Updating: Coin Fairness')
plt.legend()
plt.show()

# Posterior mean = (α + k) / (α + β + n) = (2+7)/(2+2+10) = 0.643
# MLE = k/n = 0.7
# The posterior shrinks the MLE toward the prior mean (0.5)
```

**Normal-Normal conjugacy:**

Prior: $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$

Likelihood: $x_i | \mu \sim \mathcal{N}(\mu, \sigma^2)$ (known $\sigma^2$)

Posterior: $\mu | \mathcal{D} \sim \mathcal{N}(\mu_n, \sigma_n^2)$ where:

$$\mu_n = \frac{\sigma^2 \mu_0 + n\sigma_0^2 \bar{x}}{\sigma^2 + n\sigma_0^2}, \quad \sigma_n^2 = \frac{\sigma^2 \sigma_0^2}{\sigma^2 + n\sigma_0^2}$$

The posterior mean is a **precision-weighted average** of the prior mean and the data mean. As $n \to \infty$, the posterior concentrates around the MLE.

**Dirichlet-Categorical conjugacy:**

Prior: $\mathbf{p} \sim \text{Dir}(\boldsymbol{\alpha})$

Data: counts $c_1, \ldots, c_K$ from Categorical observations

Posterior: $\mathbf{p}|\mathcal{D} \sim \text{Dir}(\alpha_1 + c_1, \ldots, \alpha_K + c_K)$

This is the foundation of Bayesian topic modeling (LDA).

### 4.9.3 Markov Chain Monte Carlo (MCMC)

When the posterior is analytically intractable (which is the case for almost all interesting models), we resort to **sampling** methods. MCMC constructs a Markov chain whose stationary distribution is the target posterior.

**Metropolis-Hastings Algorithm:**

1. Start at some $\theta_0$.
2. Propose $\theta' \sim q(\theta'|\theta_t)$ (e.g., $\theta' = \theta_t + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$).
3. Compute acceptance ratio: $\alpha = \min\left(1, \frac{p(\theta'|\mathcal{D}) \cdot q(\theta_t|\theta')}{p(\theta_t|\mathcal{D}) \cdot q(\theta'|\theta_t)}\right)$.
4. Accept $\theta_{t+1} = \theta'$ with probability $\alpha$; otherwise $\theta_{t+1} = \theta_t$.
5. After a burn-in period, the samples $\{\theta_t\}$ are (approximately) from the posterior.

Note: we only need the posterior up to a normalizing constant ($p(\theta|\mathcal{D}) \propto p(\mathcal{D}|\theta)p(\theta)$), because the evidence cancels in the ratio.

```python
def metropolis_hastings(log_posterior, theta_init, n_samples=10000,
                        proposal_std=0.1, burn_in=1000):
    """Metropolis-Hastings MCMC sampler."""
    theta = theta_init.copy()
    samples = []
    n_accepted = 0

    for i in range(n_samples + burn_in):
        # Propose
        theta_proposal = theta + np.random.normal(0, proposal_std, size=theta.shape)

        # Acceptance ratio (in log space for numerical stability)
        log_alpha = log_posterior(theta_proposal) - log_posterior(theta)
        log_u = np.log(np.random.uniform())

        if log_u < log_alpha:
            theta = theta_proposal
            n_accepted += 1

        if i >= burn_in:
            samples.append(theta.copy())

    acceptance_rate = n_accepted / (n_samples + burn_in)
    return np.array(samples), acceptance_rate

# Example: infer mean of a Gaussian with known variance
true_mu = 3.0
sigma = 1.0
data = np.random.normal(true_mu, sigma, size=50)

def log_posterior(mu):
    """Log posterior ∝ log likelihood + log prior."""
    mu = mu[0]
    # Likelihood
    log_lik = -0.5 * np.sum((data - mu)**2) / sigma**2
    # Prior: N(0, 10^2)
    log_prior = -0.5 * mu**2 / 100
    return log_lik + log_prior

samples, acc_rate = metropolis_hastings(
    log_posterior, theta_init=np.array([0.0]),
    n_samples=10000, proposal_std=0.3
)
print(f"Posterior mean: {samples.mean():.3f} (true: {true_mu})")
print(f"Posterior std: {samples.std():.3f}")
print(f"Acceptance rate: {acc_rate:.2f}")
```

**Gibbs Sampling:**

A special case of MCMC where each parameter is sampled from its full conditional distribution (the conditional distribution given all other parameters and the data). It avoids the need for a proposal distribution and has an acceptance rate of 1.

For a model with parameters $\theta_1, \theta_2, \ldots, \theta_d$:
1. Sample $\theta_1^{(t+1)} \sim p(\theta_1 | \theta_2^{(t)}, \ldots, \theta_d^{(t)}, \mathcal{D})$
2. Sample $\theta_2^{(t+1)} \sim p(\theta_2 | \theta_1^{(t+1)}, \theta_3^{(t)}, \ldots, \theta_d^{(t)}, \mathcal{D})$
3. ... and so on for each dimension.

Gibbs sampling is used in Latent Dirichlet Allocation and other Bayesian models where conditional distributions are tractable.

---

## 4.10 Probabilistic Graphical Models

Probabilistic graphical models (PGMs) use graph structures to represent conditional independence relationships among random variables. They provide a principled framework for modeling complex joint distributions.

### 4.10.1 Bayesian Networks (Directed Graphical Models)

A Bayesian network is a directed acyclic graph (DAG) where each node represents a random variable and edges represent direct dependencies. The joint distribution factorizes according to the graph structure:

$$p(x_1, x_2, \ldots, x_n) = \prod_{i=1}^{n} p(x_i | \text{parents}(x_i))$$

**Example: Naive Bayes Classifier**

The simplest Bayesian network for classification. It assumes that all features are conditionally independent given the class:

$$p(y, x_1, \ldots, x_d) = p(y) \prod_{j=1}^{d} p(x_j | y)$$

Despite this strong (and usually wrong) independence assumption, Naive Bayes works surprisingly well in practice (especially for text classification).

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# Gaussian Naive Bayes (continuous features)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
accuracy = gnb.score(X_test, y_test)

# Multinomial Naive Bayes (count features — great for text)
# Bag-of-words text classification
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=10000)
X_train_bow = vectorizer.fit_transform(train_texts)
X_test_bow = vectorizer.transform(test_texts)

mnb = MultinomialNB(alpha=1.0)  # alpha: Laplace smoothing (= Dirichlet prior)
mnb.fit(X_train_bow, train_labels)
```

### 4.10.2 Hidden Markov Models (HMMs)

An HMM models a sequence of observations generated by a hidden (unobserved) Markov chain. It has three components:
1. **Initial state distribution** $\boldsymbol{\pi}$: $\pi_i = P(z_1 = i)$
2. **Transition matrix** $\mathbf{A}$: $A_{ij} = P(z_{t+1} = j | z_t = i)$
3. **Emission distribution** $\mathbf{B}$: $b_i(x) = P(x_t | z_t = i)$

The joint probability:

$$p(\mathbf{x}, \mathbf{z}) = \pi_{z_1} \prod_{t=2}^{T} A_{z_{t-1}, z_t} \prod_{t=1}^{T} b_{z_t}(x_t)$$

**Three classical problems:**
1. **Evaluation** ($P(\mathbf{x}|\lambda)$): Forward algorithm — $O(TK^2)$
2. **Decoding** (most likely state sequence): Viterbi algorithm — $O(TK^2)$
3. **Learning** (estimate $\lambda = (\boldsymbol{\pi}, \mathbf{A}, \mathbf{B})$): Baum-Welch (EM) algorithm

**In ML:** HMMs were the dominant approach for speech recognition before deep learning, and they remain important in computational biology (gene finding), finance (regime detection), and NLP (part-of-speech tagging).

```python
from hmmlearn import hmm

# Fit a Gaussian HMM with 3 hidden states
model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
model.fit(observations)

# Decode: find most likely hidden state sequence
hidden_states = model.predict(observations)

# Score: log-likelihood
log_likelihood = model.score(observations)
```

### 4.10.3 Conditional Random Fields (CRFs)

CRFs are **undirected** graphical models for structured prediction (labeling sequences). Unlike HMMs, CRFs model $p(\mathbf{y}|\mathbf{x})$ directly (discriminative) rather than $p(\mathbf{x}, \mathbf{y})$ (generative), and they can incorporate arbitrary features of the input.

For a linear-chain CRF:

$$p(\mathbf{y}|\mathbf{x}) = \frac{1}{Z(\mathbf{x})} \exp\left(\sum_{t=1}^{T} \sum_k \lambda_k f_k(y_{t-1}, y_t, \mathbf{x}, t)\right)$$

where $f_k$ are feature functions, $\lambda_k$ are learned weights, and $Z(\mathbf{x})$ is the partition function (normalizing constant).

**In ML:** CRFs are used in named entity recognition (NER), part-of-speech tagging, and as the output layer of BiLSTM-CRF models (Lample et al., 2016). The CRF layer captures label dependencies (e.g., "I-PER" cannot follow "B-ORG").

```python
# BiLSTM-CRF in PyTorch (sketch)
import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                           bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, x, tags=None, mask=None):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)

        if tags is not None:
            # Training: return negative log-likelihood
            return -self.crf(emissions, tags, mask=mask)
        else:
            # Inference: Viterbi decoding
            return self.crf.decode(emissions, mask=mask)
```

---

## 4.11 Statistical Foundations of ML Loss Functions

This section ties together the probabilistic foundations we have developed, showing that common ML loss functions are not arbitrary choices but direct consequences of statistical principles.

### 4.11.1 Why Cross-Entropy for Classification?

**From MLE:** If $y \sim \text{Categorical}(f_\theta(x))$, the NLL is:

$$-\log p(y|x, \theta) = -\log f_\theta(x)_y = -\sum_k y_k \log f_\theta(x)_k$$

This is the cross-entropy between the one-hot label and the predicted distribution.

**From Information Theory:** Cross-entropy $H(p, q)$ measures the expected number of bits needed to encode samples from $p$ using the code optimized for $q$. Minimizing it makes $q$ as close as possible to $p$.

**From KL Divergence:** Since $H(p, q) = H(p) + D_{\text{KL}}(p \| q)$ and $H(p)$ is constant, minimizing cross-entropy is equivalent to minimizing KL divergence.

### 4.11.2 Why MSE for Regression?

**From MLE with Gaussian noise:** If $y = f_\theta(x) + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$, then minimizing the NLL gives:

$$\text{NLL} \propto \frac{1}{2\sigma^2}\sum_i (y_i - f_\theta(x_i))^2 = \frac{n}{2\sigma^2} \cdot \text{MSE}$$

**When MSE is inappropriate:** If the noise is not Gaussian (e.g., heavy-tailed errors, outliers), MSE gives too much weight to extreme errors. Alternatives:
- **MAE** (L1 loss): assumes Laplace noise, more robust to outliers.
- **Huber loss**: quadratic for small errors, linear for large errors — a smooth compromise.
- **Quantile loss**: for predicting specific quantiles rather than the mean.

### 4.11.3 Why Binary Cross-Entropy with Sigmoid?

The sigmoid function $\sigma(z) = \frac{1}{1+e^{-z}}$ is the **canonical link function** for the Bernoulli distribution in the exponential family. The pairing of sigmoid + BCE gives gradients that are simply the prediction error $\hat{y} - y$, avoiding the gradient saturation that would occur with sigmoid + MSE.

### 4.11.4 Loss Functions Summary

| Loss Function | Distribution Assumption | Use Case |
|---|---|---|
| MSE | Gaussian noise | Regression |
| MAE | Laplace noise | Robust regression |
| Cross-Entropy | Categorical | Classification |
| Binary CE | Bernoulli | Binary classification |
| Poisson NLL | Poisson | Count regression |
| Huber | Gaussian + Laplace mixture | Robust regression |
| KL Divergence | — | Distribution matching (VAE, distillation) |

---

## Exercises

### Conceptual

1. Derive the posterior distribution for the parameter $p$ of a coin, given a Beta(1, 1) prior (uniform) and observing 15 heads and 5 tails. What is the posterior mean? How does it compare to the MLE?

2. Prove that minimizing binary cross-entropy is equivalent to maximizing the likelihood of a Bernoulli model. Start from the Bernoulli PMF and derive the NLL.

3. Show that the KL divergence $D_{\text{KL}}(p \| q) \geq 0$ using Jensen's inequality. Under what condition does equality hold?

4. Explain why the Gaussian distribution maximizes entropy among all distributions with a given mean and variance. What does this imply about using MSE loss?

5. A rare disease affects 1 in 10,000 people. A test has 99.5% sensitivity and 99.9% specificity. If you test positive, what is the probability you have the disease? Explain why even highly accurate tests can have low positive predictive value for rare conditions.

### Programming

6. Implement a Naive Bayes classifier from scratch (no sklearn). Train it on the 20 Newsgroups dataset and compute accuracy. Compare with `MultinomialNB`.

7. Implement Metropolis-Hastings MCMC to infer the parameters $(\mu, \sigma^2)$ of a Gaussian from 100 data points. Use conjugate priors and verify that the MCMC posterior matches the analytical posterior. Plot the posterior samples and the analytical distribution.

8. Compute the PSI (Population Stability Index) for each feature between a training dataset and a simulated "production" dataset where some features have drifted. Visualize the PSI values and flag drifted features.

9. Implement the forward algorithm for an HMM with $K = 3$ hidden states and Gaussian emissions. Generate data from the HMM and verify that the forward algorithm computes the correct likelihood.

10. Using PyTorch, show empirically that:
    - MSE loss + linear model = least squares regression (Gaussian MLE)
    - BCE loss + sigmoid = logistic regression (Bernoulli MLE)
    - Adding L2 weight penalty = MAP with Gaussian prior
    Train on a synthetic dataset and compare the learned parameters with the analytical MLE/MAP solutions.

---

## References

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis*, 3rd Edition. Chapman and Hall/CRC.
- Koller, D., & Friedman, N. (2009). *Probabilistic Graphical Models: Principles and Techniques*. MIT Press.
- Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., & Dyer, C. (2016). Neural Architectures for Named Entity Recognition. *NAACL*.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
- Shannon, C. E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*, 27(3), 379-423.
- StatQuest with Josh Starmer. https://statquest.org/
- Wasserman, L. (2004). *All of Statistics: A Concise Course in Statistical Inference*. Springer.
- Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. *ICLR*.
- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257-286.

---

*This concludes Part I: Foundations. With Python mastery, linear algebra, calculus and optimization, and probability and statistics in hand, you are now equipped with the mathematical and computational foundations upon which all of machine learning is built. In Part II, we turn to the classical machine learning algorithms that put these foundations to work.*
