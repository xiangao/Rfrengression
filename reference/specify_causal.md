# Specify a Custom Causal Margin

Replaces the learned causal margin P(Y\|do(X)) with a user-specified
function. This allows generating joint data with a known causal effect
while preserving the learned marginal P(X,Z) and association structure.

## Usage

``` r
specify_causal(model, causal_fn)
```

## Arguments

- model:

  A frengression or frengression_seq object.

- causal_fn:

  A function taking (x, eta) tensors and returning a y tensor. Here x is
  the treatment (batch x x_dim) and eta is the noise input (batch x
  y_dim) drawn from N(0,I).

## Value

The model with the custom causal margin set.

## Examples

``` r
# \donttest{
  model <- frengression(x_dim = 1, y_dim = 1, z_dim = 1)
#> Error: Lantern is not loaded. Please use `install_torch()` to install additional dependencies.
  # Linear causal effect: Y = 5*X + noise
  model <- specify_causal(model, function(x, eta) 5 * x + eta)
#> Error: object 'model' not found
# }
```
