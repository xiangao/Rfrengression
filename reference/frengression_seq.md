# Create a FrengressionSeq Model for Longitudinal Data

Constructs a frengression model for time-varying/longitudinal causal
data. Maintains lists of T networks for each component, with sequential
dependence across time steps.

## Usage

``` r
frengression_seq(
  x_dim,
  y_dim,
  z_dim,
  T_steps,
  s_dim,
  num_layer = 3,
  hidden_dim = 100,
  noise_dim = 10,
  x_binary = FALSE,
  z_binary = FALSE,
  y_binary = FALSE,
  s_binary_dims = 0,
  s_in_predict = TRUE
)
```

## Arguments

- x_dim:

  Dimension of treatment at each time step.

- y_dim:

  Dimension of outcome at each time step.

- z_dim:

  Dimension of time-varying confounder at each time step.

- T_steps:

  Number of time steps.

- s_dim:

  Dimension of static baseline variables.

- num_layer:

  Number of layers (default: 3).

- hidden_dim:

  Hidden layer width (default: 100).

- noise_dim:

  Noise dimension (default: 10).

- x_binary:

  Logical; is X binary? (default: FALSE).

- z_binary:

  Logical; is Z binary? (default: FALSE).

- y_binary:

  Logical; is Y binary? (default: FALSE).

- s_binary_dims:

  Number of binary baseline dimensions (default: 0).

- s_in_predict:

  Include S in prediction models (default: TRUE).

## Value

A frengression_seq object.
