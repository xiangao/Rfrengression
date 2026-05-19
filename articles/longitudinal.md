# Longitudinal Data with FrengressionSeq

## Overview

[`frengression_seq()`](https://xiangao.github.io/Rfrengression/reference/frengression_seq.md)
extends the base model to time-varying (longitudinal) data. It maintains
separate networks for each time step, modeling the sequential
dependencies: $`S \to (X_0, Z_0) \to (X_1, Z_1) \to \cdots`$.

## Synthetic Longitudinal DGP

We generate 3-step longitudinal data with:

- $`S`$: binary baseline covariate
- $`X_t`$: binary treatment at time $`t`$
- $`Z_t`$: continuous time-varying confounder
- $`Y_t`$: binary outcome at time $`t`$

``` r

library(Rfrengression)
set.seed(42)

n <- 3000
T_steps <- 3

# Baseline
s <- matrix(rbinom(n, 1, 0.5), ncol = 1)

# Pre-allocate
x_all <- matrix(0, n, T_steps)
z_all <- matrix(0, n, T_steps)
y_all <- matrix(0, n, T_steps)

# Time 0
z_all[, 1] <- rnorm(n, mean = 0.3 * s)
x_all[, 1] <- rbinom(n, 1, plogis(0.5 * z_all[, 1] + 0.2 * s))
y_all[, 1] <- rbinom(n, 1, plogis(-2 + 0.5 * x_all[, 1] + 0.3 * z_all[, 1]))

# Time 1 and 2
for (t in 2:T_steps) {
  z_all[, t] <- rnorm(n, mean = 0.3 * s + 0.2 * z_all[, t - 1])
  x_all[, t] <- rbinom(n, 1, plogis(0.5 * z_all[, t] + 0.2 * s))
  y_all[, t] <- rbinom(n, 1, plogis(-2 + 0.5 * x_all[, t] +
                                        0.3 * z_all[, t] + 0.1 * x_all[, t - 1]))
}

cat("Event rates per time step:", colMeans(y_all), "\n")
#> Event rates per time step: 0.16 0.172 0.189
cat("Treatment rates per time step:", colMeans(x_all), "\n")
#> Treatment rates per time step: 0.5313333 0.5453333 0.5473333
```

## Train FrengressionSeq

Training has three stages: (1) train_xz, (2) train_e, (3) train_y.

``` r

model <- frengression_seq(
  x_dim = 1, y_dim = 1, z_dim = 1,
  T_steps = 3, s_dim = 1,
  noise_dim = 5, num_layer = 3, hidden_dim = 50,
  x_binary = TRUE, y_binary = TRUE,
  s_in_predict = TRUE
)

print(model)

# Stage 1: Marginal (X, Z) model
model <- train_xz(model, s = s, x = x_all, z = z_all,
                   num_iters = 500, lr = 1e-4, print_every = 100)

# Stage 2: Confounder model E
model <- train_e(model, s = s, x = x_all, z = z_all,
                  num_iters = 500, lr = 1e-4, print_every = 100)

# Stage 3: Outcome model Y
model <- train_y(model, s = s, x = x_all, z = z_all, y = y_all,
                  num_iters = 500, lr = 1e-4, print_every = 100)
```

## Generate Joint Samples

``` r

joint <- sample_joint(model, s = s[1:500, , drop = FALSE])

cat("Generated sample dimensions:\n")
cat("  X:", dim(joint$x), "\n")
cat("  Z:", dim(joint$z), "\n")
cat("  Y:", dim(joint$y), "\n")

cat("\nGenerated event rates:", colMeans(joint$y), "\n")
cat("Original event rates:", colMeans(y_all), "\n")
cat("\nGenerated treatment rates:", colMeans(joint$x), "\n")
cat("Original treatment rates:", colMeans(x_all), "\n")
```

## Counterfactual: Always Treated vs Never Treated

``` r

# Predict under always-treated and never-treated regimes
s_test <- matrix(rep(c(0, 1), each = 500), ncol = 1)

x_always <- matrix(1, nrow = nrow(s_test), ncol = T_steps)
x_never <- matrix(0, nrow = nrow(s_test), ncol = T_steps)

y_always <- predict_causal(model, s = s_test, x = x_always,
                            target = "mean", sample_size = 200)
y_never <- predict_causal(model, s = s_test, x = x_never,
                           target = "mean", sample_size = 200)

cat("\nE[Y_t | do(X=always 1)] per time step:\n")
for (t in seq_len(T_steps)) cat("  t =", t, ":", mean(y_always[[t]]), "\n")
cat("E[Y_t | do(X=always 0)] per time step:\n")
for (t in seq_len(T_steps)) cat("  t =", t, ":", mean(y_never[[t]]), "\n")
```

``` r

ate_t <- sapply(seq_len(T_steps), function(t) {
  mean(y_always[[t]]) - mean(y_never[[t]])
})

plot(1:T_steps, ate_t, type = "b", pch = 19, lwd = 2,
     xlab = "Time Step", ylab = "ATE",
     main = "Time-Varying Average Treatment Effect")
abline(h = 0, lty = 2, col = "gray")
```
