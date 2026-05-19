# Sample from the Causal Margin P(Y \| do(X))

Generates outcome samples under an intervention do(X = x), marginalizing
over the noise distribution.

## Usage

``` r
# S3 method for class 'frengression_seq'
sample_causal_margin(model, s, x, sample_size = 100, ...)

# S3 method for class 'frengression_surv'
sample_causal_margin(model, s, x, sample_size = 100, ...)

sample_causal_margin(model, ...)

# S3 method for class 'frengression'
sample_causal_margin(model, x, sample_size = 100, ...)
```

## Arguments

- model:

  A frengression, frengression_seq, or frengression_surv object.

- s:

  Static baseline variables (matrix or tensor).

- x:

  Intervention values for X (matrix or vector).

- sample_size:

  Number of samples per intervention value (default: 100).

- ...:

  Additional arguments passed to methods.

## Value

A matrix of outcome samples (sample_size x y_dim).
