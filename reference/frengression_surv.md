# Create a FrengressionSurv Model for Survival Data

Constructs a frengression model for survival/time-to-event data. Similar
to FrengressionSeq but with binary outcomes (events) and censoring
handling: values after the first event are masked during training.

## Usage

``` r
frengression_surv(
  x_dim,
  y_dim = 1,
  z_dim,
  T_steps,
  s_dim,
  num_layer = 3,
  hidden_dim = 100,
  noise_dim = 10,
  x_binary = FALSE,
  z_binary = FALSE,
  y_binary = TRUE,
  s_in_predict = TRUE
)
```

## Arguments

- x_dim:

  Dimension of treatment at each time step.

- y_dim:

  Dimension of outcome at each time step (default: 1).

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

  Logical; is Y binary? (default: TRUE for survival).

- s_in_predict:

  Include S in prediction models (default: TRUE).

## Value

A frengression_surv object.
