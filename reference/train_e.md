# Train the Confounder Model (E)

Trains model_e to learn P(Z \| pa(X)) for the sequential model.

## Usage

``` r
train_e(model, ...)

# S3 method for class 'frengression_seq'
train_e(
  model,
  s,
  x,
  z,
  num_iters = 100,
  lr = 0.001,
  print_every = 10,
  silent = FALSE,
  ...
)

# S3 method for class 'frengression_surv'
train_e(
  model,
  s,
  x,
  z,
  y,
  num_iters = 100,
  lr = 0.001,
  print_every = 10,
  silent = FALSE,
  ...
)
```

## Arguments

- model:

  A frengression_seq or frengression_surv object.

- ...:

  Additional arguments passed to methods.

- s:

  Static baseline variables (matrix or tensor).

- x:

  Treatment matrix or tensor.

- z:

  Confounder matrix or tensor.

- num_iters:

  Number of training iterations (default: 100).

- lr:

  Learning rate (default: 1e-3).

- print_every:

  Number of iterations between progress prints (default: 10).

- silent:

  Suppress output (default: FALSE).

- y:

  Outcome matrix or tensor (events for censoring mask).

## Value

The model with updated weights.
