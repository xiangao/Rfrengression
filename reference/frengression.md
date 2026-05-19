# Create a Frengression Model

Constructs a frengression model for cross-sectional causal data
simulation. The model learns the joint distribution P(X, Y, Z)
decomposed into three sub-networks: the marginal P(X,Z), the causal
margin P(Y\|do(X)), and the association remainder eta(X,Z).

## Usage

``` r
frengression(
  x_dim,
  y_dim,
  z_dim,
  num_layer = 3,
  hidden_dim = 100,
  noise_dim = 10,
  x_binary = FALSE,
  z_binary_dims = 0,
  y_binary = FALSE
)
```

## Arguments

- x_dim:

  Dimension of treatment variable X.

- y_dim:

  Dimension of outcome variable Y.

- z_dim:

  Dimension of confounder variable Z.

- num_layer:

  Number of layers in each sub-network (default: 3).

- hidden_dim:

  Hidden layer width (default: 100).

- noise_dim:

  Noise dimension for stochastic generation (default: 10).

- x_binary:

  Logical; is X binary? (default: FALSE).

- z_binary_dims:

  Number of leading Z dimensions that are binary (default: 0).

- y_binary:

  Logical; is Y binary? (default: FALSE).

## Value

A frengression object (S3 class) containing three sub-networks and model
parameters.

## Examples

``` r
# \donttest{
  model <- frengression(x_dim = 1, y_dim = 1, z_dim = 1)
#> Error: Lantern is not loaded. Please use `install_torch()` to install additional dependencies.
# }
```
