# Train the Outcome Model (Y)

Jointly trains model_y (causal margin) and model_eta (association
remainder). Uses the Z-permutation trick to ensure eta becomes
marginally N(0,I).

## Usage

``` r
train_y(model, ...)

# S3 method for class 'frengression'
train_y(
  model,
  x,
  z,
  y,
  num_iters = 100,
  lr = 0.001,
  print_every = 10,
  tol = 0.01,
  silent = FALSE,
  ...
)

# S3 method for class 'frengression_seq'
train_y(
  model,
  s,
  x,
  z,
  y,
  num_iters = 100,
  lr = 1e-04,
  print_every = 10,
  silent = FALSE,
  ...
)

# S3 method for class 'frengression_surv'
train_y(
  model,
  s,
  x,
  z,
  y,
  num_iters = 100,
  lr = 0.001,
  print_every = 10,
  reg_lambda = 0,
  silent = FALSE,
  ...
)
```

## Arguments

- model:

  A frengression, frengression_seq, or frengression_surv object.

- ...:

  Additional arguments passed to methods.

- x:

  Treatment matrix or tensor.

- z:

  Confounder matrix or tensor.

- y:

  Outcome matrix or tensor (events).

- num_iters:

  Number of training iterations (default: 100).

- lr:

  Learning rate (default: 1e-3).

- print_every:

  Number of iterations between progress prints (default: 10).

- tol:

  Early stopping tolerance (default: 0.01).

- silent:

  Suppress output (default: FALSE).

- s:

  Static baseline variables (matrix or tensor).

- reg_lambda:

  Regularization weight for marginal loss (default: 0).

## Value

The model with updated weights and training loss history.
