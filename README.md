# Rfrengression

[![pkgdown](https://img.shields.io/badge/pkgdown-site-blue.svg)](https://xiangao.github.io/Rfrengression/)

`Rfrengression` is an R implementation of frengression: a way to learn a joint
distribution for treatments, outcomes, and covariates, and then sample from
interventions. In practice I use `sample_causal_margin()` for `do(X=x)` draws
and `specify_causal()` when I want to replace the causal margin by a known
mechanism.

Based on [Shen & Meinshausen (2025), arXiv:2508.01018](https://arxiv.org/abs/2508.01018).

## Installation

```r
# Install torch first
install.packages("torch")
torch::install_torch()

# Install Rfrengression from source
install.packages("Rfrengression", repos = NULL, type = "source")
```

## Quick Start

```r
library(Rfrengression)

# Simulate data: Z -> X -> Y with ATE = 2
n <- 2000
z <- matrix(rnorm(n), ncol = 1)
x <- z + matrix(rnorm(n, sd = 0.5), ncol = 1)
y <- 2 * x + z + matrix(rnorm(n, sd = 0.5), ncol = 1)

# Create and train model
model <- frengression(x_dim = 1, y_dim = 1, z_dim = 1)
model <- train_y(model, x, z, y, num_iters = 500, lr = 1e-3)
model <- train_xz(model, x, z, num_iters = 500, lr = 1e-4)

# Estimate causal effect E[Y | do(X=x)]
x_grid <- matrix(seq(-2, 2, length.out = 50), ncol = 1)
y_pred <- predict(model, x_grid, type = "mean", nsample = 500)

# Generate joint samples
joint <- sample_joint(model, sample_size = 1000)

# Replace causal margin with custom function
model <- specify_causal(model, function(x, eta) 5 * x + eta)
joint_new <- sample_joint(model, sample_size = 1000)
```

## Three Model Classes

| Class | Use Case | Constructor |
|-------|----------|-------------|
| `frengression()` | Cross-sectional (continuous/binary) | `frengression(x_dim, y_dim, z_dim)` |
| `frengression_seq()` | Longitudinal / time-varying | `frengression_seq(x_dim, y_dim, z_dim, T_steps, s_dim)` |
| `frengression_surv()` | Survival / time-to-event | `frengression_surv(x_dim, y_dim, z_dim, T_steps, s_dim)` |

## Main functions

- `train_y()` — Train outcome model (model_y + model_eta)
- `train_xz()` — Train marginal model P(X, Z)
- `train_e()` — Train confounder model (sequential/survival only)
- `sample_joint()` — Sample (X, Y, Z) from learned joint
- `sample_xz()` — Sample (X, Z) from marginal
- `sample_causal_margin()` — Sample Y | do(X=x)
- `predict()` — Mean/quantile/sample predictions of causal effect
- `specify_causal()` — Replace causal margin with custom function
- `reset_y_models()` — Reinitialize outcome networks

## Vignettes

Full documentation: **<https://xiangao.github.io/Rfrengression/>**

| Vignette | Description |
|----------|-------------|
| [Continuous](https://xiangao.github.io/Rfrengression/articles/continuous.html) | Continuous treatment ADRF estimation |
| [Binary](https://xiangao.github.io/Rfrengression/articles/binary.html) | Binary treatment ATE estimation |
| [Distributional](https://xiangao.github.io/Rfrengression/articles/distributional.html) | Specifying custom causal margins |
| [Longitudinal](https://xiangao.github.io/Rfrengression/articles/longitudinal.html) | Time-varying data with `FrengressionSeq` |
| [Reference index](https://xiangao.github.io/Rfrengression/reference/index.html) | All documented functions on one page |

## Dependencies

- R >= 4.0
- [torch](https://torch.mlverse.org/) (R package)
