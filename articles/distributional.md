# Specifying Custom Causal Margins

## Overview

A key feature of frengression is
[`specify_causal()`](https://xiangao.github.io/Rfrengression/reference/specify_causal.md):
you can replace the learned causal margin with any user-defined
function. This allows generating joint data (X, Y, Z) where the causal
mechanism Y\|do(X) is exactly what you specify, while preserving the
learned marginal P(X, Z) and confounding structure.

This is useful for benchmarking causal inference methods under
controlled ground truth.

## Step 1: Train on Observed Data

``` r

library(Rfrengression)
set.seed(42)

n <- 2000
z <- matrix(rnorm(n), ncol = 1)
x <- z + matrix(rnorm(n, sd = 0.5), ncol = 1)
y <- 2 * x + z^2 + matrix(rnorm(n, sd = 0.5), ncol = 1)
```

``` r

model <- frengression(x_dim = 1, y_dim = 1, z_dim = 1, noise_dim = 5)
model <- train_y(model, x, z, y, num_iters = 300, lr = 1e-3,
                 print_every = 100, silent = TRUE)
model <- train_xz(model, x, z, num_iters = 300, lr = 1e-4,
                   print_every = 100, silent = TRUE)
```

## Step 2: Specify Different Causal Margins

The causal function takes `(x, eta)` where `eta ~ N(0, I)`:

``` r

# Causal margin 1: Linear Y = 5X + noise
model_linear <- specify_causal(model, function(x, eta) 5 * x + eta)

# Causal margin 2: Quadratic Y = X^2 + 0.5*noise
model_quad <- specify_causal(model, function(x, eta) x^2 + 0.5 * eta)

# Causal margin 3: Heterogeneous effect
model_hetero <- specify_causal(model, function(x, eta) {
  torch::torch_where(x > 0, 3 * x + eta, -x + 0.5 * eta)
})
```

## Step 3: Generate Joint Data with Each Margin

``` r

joint_linear <- sample_joint(model_linear, sample_size = 1000)
joint_quad <- sample_joint(model_quad, sample_size = 1000)
joint_hetero <- sample_joint(model_hetero, sample_size = 1000)

cat("All share the same marginal P(X,Z):\n")
cat("  Linear:  mean(X) =", round(mean(joint_linear$x), 2),
    " mean(Z) =", round(mean(joint_linear$z), 2), "\n")
cat("  Quad:    mean(X) =", round(mean(joint_quad$x), 2),
    " mean(Z) =", round(mean(joint_quad$z), 2), "\n")
cat("  Hetero:  mean(X) =", round(mean(joint_hetero$x), 2),
    " mean(Z) =", round(mean(joint_hetero$z), 2), "\n")
```

## Step 4: Verify Causal Effects

``` r

x_grid <- matrix(seq(-3, 3, length.out = 50), ncol = 1)

# Sample causal margin for each model
y_lin <- predict(model_linear, x_grid, type = "mean", nsample = 300)
y_quad <- predict(model_quad, x_grid, type = "mean", nsample = 300)
y_het <- predict(model_hetero, x_grid, type = "mean", nsample = 300)
```

``` r

par(mfrow = c(1, 3))

plot(x_grid, y_lin, type = "l", lwd = 2, col = "blue",
     main = "Linear: Y = 5X + noise", xlab = "X", ylab = "E[Y|do(X)]")
lines(x_grid, 5 * x_grid, lty = 2, col = "red", lwd = 2)
legend("topleft", c("Estimated", "True"), col = c("blue", "red"), lty = 1:2)

plot(x_grid, y_quad, type = "l", lwd = 2, col = "blue",
     main = "Quadratic: Y = X^2 + noise", xlab = "X", ylab = "E[Y|do(X)]")
lines(x_grid, x_grid^2, lty = 2, col = "red", lwd = 2)
legend("topleft", c("Estimated", "True"), col = c("blue", "red"), lty = 1:2)

plot(x_grid, y_het, type = "l", lwd = 2, col = "blue",
     main = "Heterogeneous effect", xlab = "X", ylab = "E[Y|do(X)]")
true_het <- ifelse(x_grid > 0, 3 * x_grid, -x_grid)
lines(x_grid, true_het, lty = 2, col = "red", lwd = 2)
legend("topleft", c("Estimated", "True"), col = c("blue", "red"), lty = 1:2)

par(mfrow = c(1, 1))
```

## Confounding is Preserved

Even though the causal margin changes, the confounding structure (the
association between X and Z) remains intact:

``` r

cat("Cor(X, Z) in training data:", cor(x, z), "\n")
cat("Cor(X, Z) in linear model:", cor(joint_linear$x, joint_linear$z), "\n")
cat("Cor(X, Z) in quad model:", cor(joint_quad$x, joint_quad$z), "\n")
```
