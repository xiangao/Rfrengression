# Sample from the Joint Distribution

Generates samples from the learned joint distribution P(X, Y, Z).

## Usage

``` r
# S3 method for class 'frengression_seq'
sample_joint(model, s, ...)

# S3 method for class 'frengression_surv'
sample_joint(model, s, ...)

sample_joint(model, ...)

# S3 method for class 'frengression'
sample_joint(model, sample_size = 100, ...)
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

A list with components x, y, z (matrices).
