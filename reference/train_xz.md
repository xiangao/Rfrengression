# Train the Marginal Model (XZ)

Trains model_xz to generate the joint marginal distribution of treatment
X and confounders Z.

## Usage

``` r
train_xz(model, ...)

# S3 method for class 'frengression'
train_xz(
  model,
  x,
  z,
  num_iters = 100,
  lr = 1e-04,
  print_every = 10,
  silent = FALSE,
  ...
)

# S3 method for class 'frengression_seq'
train_xz(
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
train_xz(
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

  A frengression, frengression_seq, or frengression_surv object.

- ...:

  Additional arguments passed to methods.

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

- s:

  Static baseline variables (matrix or tensor).

- y:

  Outcome matrix or tensor (events).

## Value

The model with updated weights and training loss history.
