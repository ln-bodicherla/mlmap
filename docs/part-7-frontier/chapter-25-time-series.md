# Chapter 25: Time Series and Forecasting

> *"The purpose of forecasting is not to predict the future, but to tell you what you need to know to take meaningful action in the present."* — Paul Saffo

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Analyze time series properties including stationarity, trends, seasonality, and autocorrelation, and apply appropriate decomposition methods.
2. Fit classical models (ARIMA, SARIMA, Exponential Smoothing, Prophet) using the Box-Jenkins methodology with proper order selection.
3. Engineer time series features (lags, rolling statistics, Fourier terms) for tree-based models and implement walk-forward validation.
4. Build deep learning forecasters using TCN, N-BEATS, N-HiTS, and PatchTST, understanding the architectural innovations of each.
5. Leverage foundation models for time series (TimesFM, Chronos, Moirai) for zero-shot and few-shot forecasting.
6. Detect anomalies in time series using statistical, machine learning, and deep learning methods.
7. Apply conformal prediction for distribution-free uncertainty quantification with coverage guarantees.
8. Handle practical forecasting challenges including multi-step prediction, probabilistic forecasting, and hierarchical reconciliation.

---

## 25.1 Time Series Fundamentals

A time series is a sequence of observations indexed by time: $\{y_t\}_{t=1}^{T}$. Unlike cross-sectional data, the temporal ordering matters — observations are dependent, and this dependence structure carries exploitable information.

### 25.1.1 Stationarity

A time series is *strictly stationary* if its joint distribution is invariant to time shifts: for all $k$, the distribution of $(y_{t_1}, \ldots, y_{t_n})$ equals that of $(y_{t_1+k}, \ldots, y_{t_n+k})$.

In practice, we work with *weak (second-order) stationarity*, which requires:
1. Constant mean: $\mathbb{E}[y_t] = \mu$ for all $t$.
2. Constant variance: $\text{Var}(y_t) = \sigma^2$ for all $t$.
3. Autocovariance depends only on lag: $\text{Cov}(y_t, y_{t+k}) = \gamma(k)$ for all $t$.

Most classical methods assume stationarity. Non-stationary series are transformed via differencing, detrending, or log-transformation before modeling.

**Testing for Stationarity.** The Augmented Dickey-Fuller (ADF) test is the standard unit root test:
- $H_0$: The series has a unit root (non-stationary).
- $H_1$: The series is stationary.
- Reject $H_0$ (conclude stationarity) when the test statistic is sufficiently negative (p-value < 0.05).

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def test_stationarity(series, name="Series"):
    """Perform Augmented Dickey-Fuller test for stationarity."""
    result = adfuller(series.dropna(), autolag='AIC')
    print(f"=== ADF Test: {name} ===")
    print(f"Test Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print(f"Lags Used: {result[2]}")
    for key, value in result[4].items():
        print(f"  Critical Value ({key}): {value:.4f}")
    if result[1] < 0.05:
        print("=> Stationary (reject H0)")
    else:
        print("=> Non-stationary (fail to reject H0)")
    return result[1] < 0.05
```

### 25.1.2 Trends

A *trend* is a long-term increase or decrease in the series. Trends violate the constant mean assumption of stationarity. Common types:
- **Linear trend:** $y_t = \beta_0 + \beta_1 t + \epsilon_t$.
- **Polynomial trend:** $y_t = \sum_{k=0}^{p} \beta_k t^k + \epsilon_t$.
- **Stochastic trend (random walk):** $y_t = y_{t-1} + \epsilon_t$. Removed by differencing: $\Delta y_t = y_t - y_{t-1}$.

### 25.1.3 Seasonality

*Seasonality* is a regular pattern that repeats at a fixed period $s$ (e.g., $s = 12$ for monthly data with annual seasonality, $s = 7$ for daily data with weekly seasonality). Seasonality can be:
- **Additive:** Constant amplitude regardless of level: $y_t = T_t + S_t + \epsilon_t$.
- **Multiplicative:** Amplitude proportional to level: $y_t = T_t \times S_t \times \epsilon_t$.

### 25.1.4 Autocorrelation

*Autocorrelation* measures the linear relationship between a series and its lagged values:

$$\rho(k) = \frac{\text{Cov}(y_t, y_{t+k})}{\text{Var}(y_t)} = \frac{\gamma(k)}{\gamma(0)}$$

**ACF (Autocorrelation Function):** $\rho(k)$ plotted against lag $k$. For a stationary AR(p) process, the ACF decays gradually (exponentially or with damped oscillations).

**PACF (Partial Autocorrelation Function):** The correlation between $y_t$ and $y_{t+k}$ after removing the linear dependence on $y_{t+1}, \ldots, y_{t+k-1}$. For an AR(p) process, the PACF cuts off sharply after lag $p$.

These functions are essential for model identification:

| Pattern | ACF | PACF |
|---------|-----|------|
| AR(p) | Gradual decay | Cuts off after lag p |
| MA(q) | Cuts off after lag q | Gradual decay |
| ARMA(p,q) | Gradual decay | Gradual decay |

### 25.1.5 Decomposition

Time series decomposition separates a series into its structural components.

**Additive decomposition:** $y_t = T_t + S_t + R_t$ (trend + seasonal + residual).

**Multiplicative decomposition:** $y_t = T_t \times S_t \times R_t$.

**STL (Seasonal and Trend decomposition using Loess).** Cleveland et al. (1990) proposed an iterative decomposition using locally weighted regression (loess). STL handles any seasonality period, is robust to outliers, and allows the seasonal component to vary over time.

```python
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt

# Decompose a time series with STL
stl = STL(series, period=12, robust=True)
result = stl.fit()

fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
result.observed.plot(ax=axes[0], title='Observed')
result.trend.plot(ax=axes[1], title='Trend')
result.seasonal.plot(ax=axes[2], title='Seasonal')
result.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
```

---

## 25.2 Classical Methods

Classical time series methods remain the foundation of forecasting. They are interpretable, fast, well-understood theoretically, and often competitive with deep learning on univariate forecasting tasks.

### 25.2.1 Autoregressive Models (AR)

An AR(p) model expresses the current value as a linear combination of $p$ past values:

$$y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p} + \epsilon_t$$

where $\epsilon_t \sim \text{WN}(0, \sigma^2)$ is white noise. The model is stationary if all roots of the characteristic polynomial $1 - \phi_1 z - \cdots - \phi_p z^p = 0$ lie outside the unit circle.

### 25.2.2 Moving Average Models (MA)

An MA(q) model expresses the current value as a linear combination of $q$ past error terms:

$$y_t = c + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \cdots + \theta_q \epsilon_{t-q}$$

MA models are always stationary (they are a finite sum of white noise terms) but are invertible only if all roots of $1 + \theta_1 z + \cdots + \theta_q z^q = 0$ lie outside the unit circle.

### 25.2.3 ARMA and ARIMA

**ARMA(p, q)** combines AR and MA components:

$$y_t = c + \sum_{i=1}^{p} \phi_i y_{t-i} + \epsilon_t + \sum_{j=1}^{q} \theta_j \epsilon_{t-j}$$

**ARIMA(p, d, q)** adds differencing to handle non-stationarity:
- $d$: Number of differences needed for stationarity.
- The model is applied to the $d$-th difference of the series.

### 25.2.4 Box-Jenkins Methodology

Box and Jenkins (1976) formalized a systematic approach to ARIMA modeling:

1. **Identification:** Determine $d$ (stationarity tests and differencing), then examine ACF/PACF of the differenced series to identify $p$ and $q$.
2. **Estimation:** Fit the ARIMA(p, d, q) model using maximum likelihood estimation.
3. **Diagnostic checking:** Verify that residuals are white noise (Ljung-Box test, residual ACF).
4. **Forecasting:** Generate predictions with confidence intervals.

**Order Selection with Information Criteria:**

$$\text{AIC} = -2 \ln(\hat{L}) + 2k, \quad \text{BIC} = -2 \ln(\hat{L}) + k \ln(n)$$

where $\hat{L}$ is the maximized likelihood, $k$ is the number of parameters, and $n$ is the sample size. BIC penalizes complexity more heavily and tends to select simpler models.

```python
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA

# Automatic ARIMA order selection
auto_model = pm.auto_arima(
    train_series,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    information_criterion='aic',
    max_p=5, max_q=5, max_d=2
)
print(auto_model.summary())

# Manual ARIMA fitting
model = ARIMA(train_series, order=(2, 1, 1))
fitted = model.fit()
print(fitted.summary())

# Forecast
forecast = fitted.forecast(steps=24)
conf_int = fitted.get_forecast(steps=24).conf_int(alpha=0.05)
```

### 25.2.5 SARIMA

SARIMA extends ARIMA with seasonal components: SARIMA$(p, d, q) \times (P, D, Q)_s$

$$(1 - \sum_{i=1}^p \phi_i L^i)(1 - \sum_{i=1}^P \Phi_i L^{si})(1 - L)^d(1 - L^s)^D y_t = (1 + \sum_{j=1}^q \theta_j L^j)(1 + \sum_{j=1}^Q \Theta_j L^{sj}) \epsilon_t$$

where $L$ is the lag operator ($L y_t = y_{t-1}$), and $(P, D, Q)_s$ are the seasonal AR, differencing, and MA orders at seasonal period $s$.

### 25.2.6 Exponential Smoothing

Exponential smoothing methods produce forecasts as weighted averages of past observations, with weights decaying exponentially.

**Simple Exponential Smoothing (SES):** For series with no trend or seasonality:

$$\hat{y}_{t+1|t} = \alpha y_t + (1 - \alpha) \hat{y}_{t|t-1}$$

where $\alpha \in (0, 1)$ is the smoothing parameter.

**Holt's Linear Method:** Adds a trend component:

$$\ell_t = \alpha y_t + (1 - \alpha)(\ell_{t-1} + b_{t-1}) \quad \text{(level)}$$
$$b_t = \beta^*(\ell_t - \ell_{t-1}) + (1 - \beta^*) b_{t-1} \quad \text{(trend)}$$
$$\hat{y}_{t+h|t} = \ell_t + h b_t$$

**Holt-Winters:** Adds seasonality (additive or multiplicative) on top of Holt's method. The ETS (Error, Trend, Seasonal) framework by Hyndman et al. (2008) provides a comprehensive taxonomy with 30 model variants and automatic selection.

### 25.2.7 Prophet

Prophet (Taylor & Letham, 2018) by Facebook/Meta is a modular decomposable time series model designed for business forecasting:

$$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$

where:
- $g(t)$: Trend (piecewise linear or logistic growth with automatic changepoint detection).
- $s(t)$: Seasonality (Fourier series with multiple seasonal periods).
- $h(t)$: Holiday effects (user-specified).

Prophet is robust to missing data, trend shifts, and outliers. It fits a Bayesian model using Stan and provides interpretable uncertainty intervals.

```python
from prophet import Prophet

# Prepare data (Prophet requires 'ds' and 'y' columns)
df = pd.DataFrame({'ds': dates, 'y': values})

# Fit Prophet
model = Prophet(
    changepoint_prior_scale=0.05,   # Flexibility of trend
    seasonality_prior_scale=10.0,   # Flexibility of seasonality
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)
model.add_country_holidays(country_name='US')
model.fit(df)

# Forecast 365 days ahead
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Visualize components
model.plot_components(forecast)
```

---

## 25.3 Machine Learning for Time Series

Tree-based models (XGBoost, LightGBM) with careful feature engineering often outperform deep learning on tabular time series data, particularly for heterogeneous features and small-to-medium datasets.

### 25.3.1 Feature Engineering

The key to using ML models for time series is transforming the forecasting problem into a supervised learning problem through feature engineering.

**Lag Features.** Past values as features: $y_{t-1}, y_{t-2}, \ldots, y_{t-k}$.

**Rolling Statistics.** Aggregations over windows: rolling mean, standard deviation, min, max, median over the last $w$ observations.

**Date Features.** Extracted from the timestamp: hour, day of week, day of month, month, quarter, year, is_weekend, is_holiday.

**Fourier Terms.** Pairs of sine and cosine terms at different frequencies capture seasonality:

$$\sin\left(\frac{2\pi k t}{P}\right), \quad \cos\left(\frac{2\pi k t}{P}\right) \quad \text{for } k = 1, 2, \ldots, K$$

where $P$ is the seasonal period. The number of terms $K$ controls the smoothness of the seasonal pattern.

```python
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

def create_time_features(df, target_col, lags=[1, 7, 14, 28],
                          windows=[7, 14, 28]):
    """Create time series features for ML models."""
    features = df.copy()

    # Lag features
    for lag in lags:
        features[f'lag_{lag}'] = features[target_col].shift(lag)

    # Rolling statistics
    for window in windows:
        rolled = features[target_col].shift(1).rolling(window)
        features[f'rolling_mean_{window}'] = rolled.mean()
        features[f'rolling_std_{window}'] = rolled.std()
        features[f'rolling_min_{window}'] = rolled.min()
        features[f'rolling_max_{window}'] = rolled.max()

    # Date features
    features['dayofweek'] = features.index.dayofweek
    features['month'] = features.index.month
    features['quarter'] = features.index.quarter
    features['dayofyear'] = features.index.dayofyear
    features['is_weekend'] = (features.index.dayofweek >= 5).astype(int)

    # Fourier terms for yearly seasonality
    for k in range(1, 4):
        features[f'sin_{k}'] = np.sin(2 * np.pi * k * features.index.dayofyear / 365.25)
        features[f'cos_{k}'] = np.cos(2 * np.pi * k * features.index.dayofyear / 365.25)

    return features.dropna()
```

### 25.3.2 Walk-Forward Validation

Standard cross-validation is invalid for time series because it violates temporal ordering — the model would train on future data and test on past data. Walk-forward (expanding window or sliding window) validation respects the temporal order:

```python
from sklearn.model_selection import TimeSeriesSplit

def walk_forward_validation(df, feature_cols, target_col, n_splits=5):
    """Time-respecting walk-forward validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        X_train = df.iloc[train_idx][feature_cols]
        y_train = df.iloc[train_idx][target_col]
        X_test = df.iloc[test_idx][feature_cols]
        y_test = df.iloc[test_idx][target_col]

        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            verbosity=-1
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        scores.append(mae)
        print(f"Fold {fold+1}: MAE = {mae:.4f}")

    print(f"\nMean MAE: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    return scores
```

---

## 25.4 Deep Learning for Time Series

Deep learning approaches to time series forecasting have matured significantly, with several architectures demonstrating strong performance.

### 25.4.1 RNNs and LSTMs

Recurrent neural networks process sequences element-by-element, maintaining a hidden state that summarizes past information:

$$\mathbf{h}_t = f(\mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{W}_x \mathbf{x}_t + \mathbf{b})$$

LSTMs (Hochreiter & Schmidhuber, 1997) address the vanishing gradient problem with a gated cell state that can store information over long horizons. For time series forecasting, a common architecture is an encoder-decoder LSTM:

1. **Encoder:** Processes the historical sequence, encoding it into a fixed-size context vector.
2. **Decoder:** Generates future predictions autoregressively, using teacher forcing during training.

While LSTMs were the dominant deep learning approach before 2020, they have largely been supplanted by more specialized architectures.

### 25.4.2 Temporal Convolutional Networks (TCN)

TCNs (Bai et al., 2018) apply 1D convolutions to sequences with two key modifications:

**Causal Convolutions.** The output at time $t$ depends only on inputs at times $\leq t$, ensuring no information leakage from the future. This is implemented by padding only the left side of the input.

**Dilated Convolutions.** To achieve large receptive fields without excessive depth or parameters, TCNs use exponentially increasing dilation rates:

$$y_t = \sum_{k=0}^{K-1} w_k \cdot x_{t - d \cdot k}$$

where $d$ is the dilation rate. With dilation rates $d = 1, 2, 4, 8, \ldots, 2^{L-1}$ across $L$ layers, the receptive field grows exponentially: $r = 2^L \cdot (K - 1) + 1$ with kernel size $K$.

The TCN architecture stacks residual blocks, each containing dilated causal convolutions, weight normalization, ReLU activations, and dropout. TCNs offer several advantages over RNNs: parallelizable training (no sequential dependency), stable gradients (no vanishing/exploding), flexible receptive field, and lower memory during training.

### 25.4.3 N-BEATS

N-BEATS (Neural Basis Expansion Analysis for Time Series) by Oreshkin et al. (2020) is a pure deep learning architecture specifically designed for time series forecasting.

**Architecture.** N-BEATS uses a deep stack of blocks, each consisting of:
1. A fully connected network that processes the lookback window.
2. Two linear layers that produce *basis expansion coefficients* for backcast and forecast.
3. The coefficients are multiplied by basis functions to produce the backcast (reconstruction of the input) and forecast (future prediction).

$$\hat{\mathbf{x}}_\ell = \sum_{k=1}^{K} \theta_k^b \cdot \mathbf{v}_k^b \quad \text{(backcast)}, \qquad \hat{\mathbf{y}}_\ell = \sum_{k=1}^{K} \theta_k^f \cdot \mathbf{v}_k^f \quad \text{(forecast)}$$

where $\theta_k$ are the learned coefficients and $\mathbf{v}_k$ are the basis vectors.

**Doubly Residual Architecture.** The input to each block is the residual from the previous block's backcast: $\mathbf{x}_{\ell+1} = \mathbf{x}_\ell - \hat{\mathbf{x}}_\ell$. The final forecast is the sum of all blocks' forecasts: $\hat{\mathbf{y}} = \sum_\ell \hat{\mathbf{y}}_\ell$.

**Interpretable Configuration.** By constraining the basis functions:
- **Trend block:** Uses polynomial basis (powers of $t$): $\mathbf{v}_k = [0^k, 1^k, \ldots, (H-1)^k]$.
- **Seasonality block:** Uses Fourier basis (sines and cosines at harmonic frequencies).

This decomposition is learned end-to-end, providing interpretable trend and seasonal components without explicit feature engineering.

### 25.4.4 N-HiTS

N-HiTS (Neural Hierarchical Interpolation for Time Series) by Challu et al. (2023) extends N-BEATS with *hierarchical interpolation* — different blocks focus on different temporal scales by downsampling the input at different rates and expressing their forecasts at different frequencies, then interpolating to the target resolution.

This multi-rate approach dramatically improves efficiency (3-5x faster than N-BEATS) and accuracy on long-horizon forecasting tasks by allowing coarse blocks to capture low-frequency trends and fine blocks to capture high-frequency patterns.

### 25.4.5 PatchTST

PatchTST (Nie et al., 2023) adapts the Vision Transformer paradigm to time series by treating time series "patches" analogously to image patches in ViT.

**Patching.** The input time series is segmented into fixed-length patches (e.g., patch length $P = 16$ with stride $S$), each of which is linearly projected into an embedding vector. This serves the same purpose as patch embedding in ViT: it reduces the sequence length (from $L$ to $\lceil L/S \rceil$) and captures local semantic information within each patch.

**Channel Independence.** Each channel (variable in multivariate time series) is processed independently through the same Transformer backbone. This counterintuitive design choice prevents overfitting to inter-channel correlations and allows the model to generalize across channels, acting as a form of regularization.

**Architecture.** Standard Transformer encoder with:
1. Patch embedding + positional encoding.
2. Multi-head self-attention across patches.
3. Feedforward layers.
4. Linear head for prediction.

PatchTST significantly outperforms prior Transformer-based forecasters (Informer, Autoformer, FEDformer) on long-term forecasting benchmarks, and its simplicity makes it a strong baseline.

```python
# PatchTST conceptual implementation
import torch
import torch.nn as nn

class PatchTST(nn.Module):
    def __init__(self, seq_len=512, pred_len=96, patch_len=16,
                 stride=8, d_model=128, nhead=4, num_layers=3,
                 n_channels=1, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_channels = n_channels
        self.pred_len = pred_len

        # Number of patches
        self.num_patches = (seq_len - patch_len) // stride + 1

        # Patch embedding
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, d_model)
        )
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm = nn.LayerNorm(d_model)

        # Prediction head: flatten + linear
        self.head = nn.Linear(self.num_patches * d_model, pred_len)

    def forward(self, x):
        # x: (batch, seq_len, n_channels)
        B, L, C = x.shape
        # Channel independence: process each channel separately
        # Reshape: (B*C, L, 1) -> process -> (B, C, pred_len)
        x = x.permute(0, 2, 1).reshape(B * C, L)  # (B*C, L)

        # Create patches
        patches = x.unfold(dimension=1, size=self.patch_len,
                          step=self.stride)  # (B*C, num_patches, patch_len)

        # Embed patches
        z = self.patch_embed(patches) + self.pos_embed  # (B*C, num_patches, d_model)
        z = self.dropout(z)

        # Transformer encoding
        z = self.transformer(z)
        z = self.norm(z)

        # Flatten and predict
        z = z.reshape(B * C, -1)  # (B*C, num_patches * d_model)
        out = self.head(z)  # (B*C, pred_len)

        return out.reshape(B, C, self.pred_len).permute(0, 2, 1)
```

---

## 25.5 Foundation Models for Time Series

Just as large language models transformed NLP and vision foundation models transformed computer vision, a new generation of *time series foundation models* is emerging — pretrained on massive, diverse time series corpora and capable of zero-shot forecasting on unseen datasets.

### 25.5.1 TimesFM

TimesFM (Das et al., 2024) by Google is a decoder-only foundation model for time series forecasting, pretrained on a large corpus of 100 billion time points from Google Trends, Wiki pageviews, and synthetic data.

**Key Design Decisions:**
- **Patched decoder:** Like PatchTST, input is divided into patches, but uses a decoder-only (causal) architecture to enable autoregressive generation.
- **Input tokenization:** Time series patches are linearly projected into tokens, avoiding explicit discretization.
- **Variable-length context:** Handles varying input lengths up to 512 time points.
- **Zero-shot capability:** Forecasts unseen time series without fine-tuning, achieving competitive or superior performance compared to supervised models trained on each dataset individually.

### 25.5.2 Chronos

Chronos (Ansari et al., 2024) by Amazon takes a radically different approach: it *tokenizes* time series values into discrete tokens and uses a language model architecture (T5-based) for forecasting.

**Tokenization Pipeline:**
1. **Scaling:** Normalize each time series by its mean absolute value to handle diverse scales.
2. **Quantization:** Map continuous values to discrete bins using fixed quantile-based bin edges.
3. **Token embedding:** Map bin indices to learned embeddings, exactly as in NLP.

**Training:** Chronos uses a standard language modeling objective — cross-entropy loss over the token vocabulary — applied to the quantized time series values. This allows direct reuse of pretrained language model architectures and training infrastructure.

**Probabilistic Forecasting:** At inference, Chronos generates multiple sample paths by sampling from the predicted token distribution at each step, naturally providing uncertainty estimates.

**Results:** Chronos models (ranging from 20M to 710M parameters) achieve strong zero-shot performance, outperforming task-specific models on many benchmarks. The largest model is particularly competitive with supervised methods that train on each dataset.

### 25.5.3 Moirai

Moirai (Woo et al., 2024) is a universal time series forecaster designed to handle the heterogeneity of real-world time series:

- **Any frequency:** Handles data at any sampling frequency (minutely to yearly) through frequency-aware tokenization.
- **Any number of variables:** Processes univariate and multivariate time series.
- **Any forecast horizon:** Flexible prediction lengths.
- **Mixture distribution output:** Predicts parameters of a mixture distribution for probabilistic forecasting.

### 25.5.4 The Paradigm Shift

The shift from task-specific to foundation models mirrors the evolution in NLP:

| Era | Approach | Example |
|-----|----------|---------|
| Classical | Task-specific statistical model | ARIMA per series |
| ML | Task-specific ML model | LightGBM per dataset |
| Global DL | One model, many series | N-BEATS trained on M4 |
| Foundation | One model, any series | Chronos zero-shot |

The key advantages of foundation models:
1. **Zero-shot forecasting:** No training data needed for new series.
2. **Transfer learning:** Pretrained knowledge transfers across domains.
3. **Reduced engineering:** No feature engineering, order selection, or hyperparameter tuning per series.
4. **Scalability:** A single model handles thousands of heterogeneous series.

However, limitations remain: foundation models may underperform specialized models on domain-specific data with strong known structure, and they require significant compute for pretraining.

---

## 25.6 Anomaly Detection in Time Series

Time series anomaly detection identifies unusual patterns that deviate from expected behavior. Applications include fraud detection, equipment failure prediction, network intrusion detection, and quality monitoring.

### 25.6.1 Statistical Methods

**Z-Score.** Flag observations where $|z_t| = |(y_t - \mu) / \sigma| > \tau$ (typically $\tau = 3$). For streaming data, use exponentially weighted moving average (EWMA) for $\mu$ and $\sigma$.

**IQR (Interquartile Range).** Observations below $Q_1 - 1.5 \cdot \text{IQR}$ or above $Q_3 + 1.5 \cdot \text{IQR}$ are flagged. Robust to outliers in the calculation itself.

**CUSUM (Cumulative Sum).** Detects shifts in the mean by accumulating deviations from a target value. Effective for detecting gradual drifts.

### 25.6.2 Machine Learning Methods

**Isolation Forest (Liu et al., 2008).** Builds an ensemble of random trees that recursively partition the feature space. Anomalies are isolated in fewer splits (closer to the root), yielding a lower anomaly score. For time series, apply to engineered features (lags, rolling statistics).

**One-Class SVM.** Learns a boundary that encloses normal data in feature space. Points outside the boundary are anomalies. The RBF kernel maps data to a higher-dimensional space where a linear boundary can be found.

### 25.6.3 Deep Learning Methods

**Autoencoder-Based.** Train an autoencoder on normal data. At inference, the reconstruction error $\|x_t - \hat{x}_t\|$ serves as an anomaly score — anomalous patterns that the autoencoder has not learned to reconstruct will have high error.

**LSTM Autoencoder (LSTM-AE).** Uses LSTM encoder and decoder to capture temporal dependencies:

```python
import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder for time series anomaly detection."""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers,
                                batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                                batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Encode
        _, (hidden, cell) = self.encoder(x)

        # Decode: repeat hidden state for each time step
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))

        # Reconstruct
        reconstruction = self.output_layer(decoder_output)
        return reconstruction

    def anomaly_score(self, x):
        """Compute per-sample reconstruction error."""
        with torch.no_grad():
            x_hat = self.forward(x)
            # Mean squared error per sample
            mse = ((x - x_hat) ** 2).mean(dim=(1, 2))
        return mse
```

**Transformer-Based Anomaly Detection.** Transformer architectures (e.g., Anomaly Transformer by Xu et al., 2022) use the association discrepancy between the series-association (learned attention) and prior-association (fixed Gaussian kernel) to detect anomalies. Points where the learned attention pattern deviates significantly from the prior are flagged as anomalous.

---

## 25.7 Conformal Prediction for Time Series

A critical question in forecasting is: *how uncertain is this prediction?* Traditional prediction intervals (e.g., from ARIMA or Gaussian assumptions) rely on distributional assumptions that are often violated. Conformal prediction provides *distribution-free* uncertainty quantification with finite-sample coverage guarantees.

### 25.7.1 Coverage Guarantee

Conformal prediction guarantees that a prediction interval $C(X_{n+1})$ contains the true value $Y_{n+1}$ with probability at least $1 - \alpha$:

$$P(Y_{n+1} \in C(X_{n+1})) \geq 1 - \alpha$$

This guarantee holds for *any* distribution (no parametric assumptions) under the mild condition of *exchangeability* (a weaker condition than i.i.d.).

### 25.7.2 Split Conformal Prediction

The simplest conformal prediction method:

1. **Split** data into training and calibration sets.
2. **Train** any point prediction model $\hat{f}$ on the training set.
3. **Compute** nonconformity scores on the calibration set: $s_i = |y_i - \hat{f}(x_i)|$ for $i = 1, \ldots, n$.
4. **Compute** the conformal quantile: $\hat{q} = \text{Quantile}_{(1-\alpha)(1 + 1/n)}(\{s_1, \ldots, s_n\})$.
5. **Predict:** $C(x_{n+1}) = [\hat{f}(x_{n+1}) - \hat{q}, \, \hat{f}(x_{n+1}) + \hat{q}]$.

This produces prediction intervals with exactly $1 - \alpha$ marginal coverage.

### 25.7.3 Conformalized Quantile Regression

Split conformal prediction produces constant-width intervals (same $\hat{q}$ for all predictions). Conformalized Quantile Regression (CQR) by Romano et al. (2019) produces *adaptive* intervals that are narrower when the model is confident and wider when it is uncertain:

1. Train a quantile regression model to predict the $\alpha/2$ and $1 - \alpha/2$ quantiles: $\hat{q}_\text{lo}(x)$ and $\hat{q}_\text{hi}(x)$.
2. Compute calibration scores: $s_i = \max(\hat{q}_\text{lo}(x_i) - y_i, \, y_i - \hat{q}_\text{hi}(x_i))$.
3. Compute the conformal quantile $\hat{q}$ from the scores.
4. Predict: $C(x) = [\hat{q}_\text{lo}(x) - \hat{q}, \, \hat{q}_\text{hi}(x) + \hat{q}]$.

### 25.7.4 Adaptive Conformal Inference for Time Series

Standard conformal prediction assumes exchangeability, which is violated by time series (observations are temporally dependent). Adaptive Conformal Inference (ACI) by Gibbs and Candes (2021) addresses this by dynamically adjusting the significance level $\alpha_t$ based on recent coverage:

$$\alpha_{t+1} = \alpha_t + \gamma(\alpha - \text{err}_t)$$

where $\text{err}_t = \mathbb{1}[y_t \notin C_t(x_t)]$ and $\gamma > 0$ is a step size. If recent intervals miss too often, $\alpha_t$ decreases (intervals widen); if they cover too much, $\alpha_t$ increases (intervals narrow). This maintains approximate long-run coverage even under distribution shift.

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

class ConformalForecaster:
    """Conformalized quantile regression for time series."""
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.q_lo_model = GradientBoostingRegressor(
            loss='quantile', alpha=alpha / 2, n_estimators=200
        )
        self.q_hi_model = GradientBoostingRegressor(
            loss='quantile', alpha=1 - alpha / 2, n_estimators=200
        )
        self.conformal_quantile = None

    def fit(self, X_train, y_train, X_cal, y_cal):
        """Fit quantile models and calibrate."""
        # Train quantile regression models
        self.q_lo_model.fit(X_train, y_train)
        self.q_hi_model.fit(X_train, y_train)

        # Calibrate on held-out calibration set
        q_lo_cal = self.q_lo_model.predict(X_cal)
        q_hi_cal = self.q_hi_model.predict(X_cal)

        # Nonconformity scores
        scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)

        # Conformal quantile (with finite-sample correction)
        n = len(scores)
        self.conformal_quantile = np.quantile(
            scores,
            min(1.0, (1 - self.alpha) * (1 + 1 / n))
        )

    def predict(self, X):
        """Predict with conformal intervals."""
        q_lo = self.q_lo_model.predict(X) - self.conformal_quantile
        q_hi = self.q_hi_model.predict(X) + self.conformal_quantile
        point = (q_lo + q_hi) / 2
        return point, q_lo, q_hi
```

---

## 25.8 Practical Considerations

### 25.8.1 Multi-Step Forecasting

Predicting multiple future steps introduces additional complexity:

**Recursive (Iterative).** Predict one step ahead, feed the prediction back as input, repeat. Error accumulates over the horizon.

**Direct.** Train a separate model for each forecast horizon $h$: $\hat{y}_{t+h} = f_h(\mathbf{x}_t)$. No error accumulation but does not capture dependencies between forecast steps, and requires $H$ models.

**MIMO (Multiple-Input Multiple-Output).** A single model outputs all forecast steps simultaneously: $(\hat{y}_{t+1}, \ldots, \hat{y}_{t+H}) = f(\mathbf{x}_t)$. Captures inter-step dependencies with a single model. Used by N-BEATS, PatchTST, and most modern architectures.

**DirRec (Hybrid).** Combines direct and recursive: predict directly but feed earlier direct predictions as features for later horizons.

### 25.8.2 Probabilistic Forecasting

Point forecasts are often insufficient — decision-makers need uncertainty estimates. Probabilistic forecasting produces full predictive distributions.

**Parametric.** Predict parameters of a distribution (e.g., $\mu, \sigma$ for Gaussian, or quantiles). The loss function matches the assumed distribution:

- **Gaussian:** Negative log-likelihood: $\mathcal{L} = \frac{1}{2} \log \sigma^2 + \frac{(y - \mu)^2}{2\sigma^2}$.
- **Quantile:** Pinball loss for quantile $q$: $\mathcal{L}_q = \max(q(y - \hat{y}_q), (q-1)(y - \hat{y}_q))$.

**Non-parametric.** Generate sample paths via Monte Carlo sampling (Chronos) or ensemble methods.

**Evaluation.** Probabilistic forecasts are evaluated using:
- **CRPS (Continuous Ranked Probability Score):** $\text{CRPS}(F, y) = \int_{-\infty}^{\infty} (F(z) - \mathbb{1}[y \leq z])^2 \, dz$.
- **Calibration:** Do $p$% prediction intervals contain the true value $p$% of the time?
- **Sharpness:** Narrower intervals are preferred (conditional on calibration).

### 25.8.3 Hierarchical Time Series

Many real-world forecasting problems involve hierarchical or grouped time series. For example, retail sales can be disaggregated by region, store, product category, and individual product. Forecasts at different levels must be *coherent* — they should add up consistently.

**Reconciliation Methods:**
- **Bottom-up:** Forecast at the most granular level and aggregate up.
- **Top-down:** Forecast at the top level and distribute down using historical proportions.
- **Optimal reconciliation (MinT):** Forecast independently at all levels, then apply a linear transformation to ensure coherence while minimizing forecast variance (Wickramasuriya et al., 2019).

The reconciliation matrix $\mathbf{P}$ maps base forecasts $\hat{\mathbf{y}}$ to coherent forecasts $\tilde{\mathbf{y}}$:

$$\tilde{\mathbf{y}} = \mathbf{S} (\mathbf{S}^T \mathbf{W}^{-1} \mathbf{S})^{-1} \mathbf{S}^T \mathbf{W}^{-1} \hat{\mathbf{y}}$$

where $\mathbf{S}$ is the summing matrix (encoding the hierarchy) and $\mathbf{W}$ is a weight matrix (e.g., the covariance matrix of base forecast errors).

---

## Exercises

### Conceptual Questions

1. **Stationarity and differencing.** Explain the difference between a trend-stationary process and a difference-stationary process. How does the appropriate detrending method differ for each? Give a real-world example of each.

2. **ACF/PACF interpretation.** A time series has PACF with significant spikes at lags 1 and 2 only, and an ACF that decays exponentially. What ARIMA order would you suggest? Justify your answer.

3. **PatchTST design.** Why does channel independence (processing each variable separately) work better than channel mixing for long-term forecasting? Under what conditions might channel dependence be important?

4. **Conformal prediction assumptions.** Explain why standard conformal prediction's exchangeability assumption is violated by time series data. How does Adaptive Conformal Inference address this?

### Implementation Exercises

5. **ARIMA pipeline.** Implement a complete Box-Jenkins pipeline for the Air Passengers dataset: test for stationarity, difference as needed, identify orders from ACF/PACF, fit ARIMA with `auto_arima`, check residual diagnostics, and forecast 24 months ahead with confidence intervals.

6. **LightGBM forecaster.** Build a LightGBM forecaster for hourly electricity demand. Engineer lag features, rolling statistics, calendar features, and Fourier terms. Evaluate using walk-forward validation with 5 folds. Compare with ARIMA and Prophet.

7. **N-BEATS implementation.** Implement the N-BEATS architecture (generic basis) from scratch in PyTorch. Train on the M4 Monthly dataset. Implement the interpretable version and visualize the learned trend and seasonal decomposition.

8. **Conformal prediction.** Apply conformalized quantile regression to a real-world forecasting task. Train LightGBM quantile regressors, calibrate using a held-out calibration set, and verify that the coverage guarantee holds on the test set. Compare interval widths with naive Gaussian intervals.

### Research Questions

9. **Foundation model evaluation.** Compare Chronos (zero-shot) versus a fine-tuned PatchTST on the ETTh1 benchmark. Under what conditions does zero-shot forecasting match or exceed supervised training? How does performance change as the amount of training data varies?

10. **Anomaly detection comparison.** Compare statistical (Z-score), ML (Isolation Forest), and deep learning (LSTM-AE) anomaly detectors on the NAB (Numenta Anomaly Benchmark) dataset. Analyze the types of anomalies each method excels at (point anomalies, contextual anomalies, collective anomalies).

---

## References

1. Ansari, A. F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., ... & Januschowski, T. (2024). Chronos: Learning the Language of Time Series. *arXiv:2403.07815*.

2. Bai, S., Kolter, J. Z., & Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. *arXiv:1803.01271*.

3. Box, G. E. P., & Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control* (revised ed.). Holden-Day.

4. Challu, C., Olivares, K. G., Oreshkin, B. N., Ramirez, F. G., Canseco, M. M., & Dubrawski, A. (2023). N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting. *AAAI*.

5. Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). STL: A Seasonal-Trend Decomposition Procedure Based on Loess. *Journal of Official Statistics*, 6(1), 3-73.

6. Das, A., Kong, W., Leber, A., Mathews, R., Mohan, R., & Rose, S. (2024). A Decoder-Only Foundation Model for Time-Series Forecasting. *ICML*.

7. Gibbs, I., & Candes, E. (2021). Adaptive Conformal Inference Under Distribution Shift. *NeurIPS*.

8. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.

9. Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008). *Forecasting with Exponential Smoothing: The State Space Approach*. Springer.

10. Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. *ICDM*.

11. Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers. *ICLR*.

12. Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2020). N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting. *ICLR*.

13. Romano, Y., Patterson, E., & Candes, E. (2019). Conformalized Quantile Regression. *NeurIPS*.

14. Taylor, S. J., & Letham, B. (2018). Forecasting at Scale. *The American Statistician*, 72(1), 37-45.

15. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.

16. Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019). Optimal Forecast Reconciliation for Hierarchical and Grouped Time Series Through Trace Minimization. *JASA*, 114(526), 804-819.

17. Woo, G., Liu, C., Kumar, A., Xiong, C., Savarese, S., & Sahoo, D. (2024). Unified Training of Universal Time Series Forecasting Transformers. *ICML*.

18. Xu, J., Wu, H., Wang, J., & Long, M. (2022). Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy. *ICLR*.
