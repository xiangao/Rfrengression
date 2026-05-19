# Continuous Treatment: Average Dose-Response Function

## Overview

This vignette demonstrates frengression for continuous treatment data.
We generate synthetic data with a known causal effect (ADRF), train a
frengression model, and verify it recovers the true dose-response curve.

## Data Generating Process

The DGP has a confounder Z that affects both treatment X and outcome Y:

- $`Z \sim N(0, 1)`$
- $`X = Z + \varepsilon_X`$, where $`\varepsilon_X \sim N(0, 0.5^2)`$
- $`Y = 2X + Z^2 + \varepsilon_Y`$, where
  $`\varepsilon_Y \sim N(0, 0.5^2)`$

The true ADRF is $`E[Y | do(X=x)] = 2x + E[Z^2] = 2x + 1`$.

``` r

library(Rfrengression)
set.seed(42)

n <- 2000
z <- matrix(rnorm(n), ncol = 1)
x <- z + matrix(rnorm(n, sd = 0.5), ncol = 1)
y <- 2 * x + z^2 + matrix(rnorm(n, sd = 0.5), ncol = 1)
```

## Train Frengression

Training is two-stage: first the outcome model (model_y + model_eta),
then the marginal model (model_xz).

``` r

model <- frengression(x_dim = 1, y_dim = 1, z_dim = 1, noise_dim = 5)

# Stage 1: Train outcome model
model <- train_y(model, x, z, y, num_iters = 500, lr = 1e-3,
                 print_every = 100)

# Stage 2: Train marginal model
model <- train_xz(model, x, z, num_iters = 500, lr = 1e-4,
                   print_every = 100)

print(model)
```

## Estimate the ADRF

We evaluate the causal effect at a grid of treatment values:

``` r

x_grid <- matrix(seq(-3, 3, length.out = 50), ncol = 1)
y_pred <- predict(model, x_grid, type = "mean", nsample = 500)

# True ADRF
y_true <- 2 * x_grid + 1

# Quantile bands
y_quant <- predict(model, x_grid, type = "quantile", nsample = 500,
                   quantiles = c(0.1, 0.9))
```

## Plot Results

``` r

plot(x_grid, y_pred, type = "l", lwd = 2, col = "blue",
     xlab = "Treatment (X)", ylab = "E[Y | do(X)]",
     main = "Average Dose-Response Function")
lines(x_grid, y_true, lty = 2, lwd = 2, col = "red")
if (is.array(y_quant) && length(dim(y_quant)) == 3) {
  lines(x_grid, y_quant[, 1, 1], lty = 3, col = "blue")
  lines(x_grid, y_quant[, 1, 2], lty = 3, col = "blue")
}
legend("topleft", c("Estimated ADRF", "True ADRF", "90% band"),
       col = c("blue", "red", "blue"), lty = c(1, 2, 3), lwd = c(2, 2, 1))
```

## Verify with Joint Sampling

``` r

joint <- sample_joint(model, sample_size = 1000)
cat("Generated joint sample dimensions:\n")
cat("  X:", dim(joint$x), "\n")
cat("  Y:", dim(joint$y), "\n")
cat("  Z:", dim(joint$z), "\n")
cat("\nCorrelation X-Z (should be ~0.89):", cor(joint$x, joint$z), "\n")
```
