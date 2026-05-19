# Sample from the Marginal Distribution of (X, Z)

Generates samples from the learned marginal distribution P(X, Z).

## Usage

``` r
# S3 method for class 'frengression_seq'
sample_xz(model, s, ...)

sample_xz(model, ...)

# S3 method for class 'frengression'
sample_xz(model, sample_size = 100, ...)
```

## Arguments

- model:

  A frengression, frengression_seq, or frengression_surv object.

- s:

  Static baseline variables (matrix or tensor).

- ...:

  Additional arguments passed to methods.

- sample_size:

  Number of samples to generate (default: 100).

## Value

A list with components x, z (matrices).
