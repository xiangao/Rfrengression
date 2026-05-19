# Predict Causal Effects (generic)

Predict Causal Effects (generic)

## Usage

``` r
# S3 method for class 'frengression_seq'
predict_causal(model, s, x, target = "mean", sample_size = 100, ...)

# S3 method for class 'frengression_surv'
predict_causal(model, s, x, sample_size = 100, ...)

predict_causal(model, ...)

# S3 method for class 'frengression'
predict_causal(model, x, target = "mean", sample_size = 100, ...)
```

## Arguments

- model:

  A frengression object.

- s:

  Static baseline variables (matrix or tensor).

- x:

  Intervention values.

- target:

  "mean" or "sample" (default: "mean").

- sample_size:

  Number of Monte Carlo samples (default: 100).

- ...:

  Additional arguments passed to methods.

## Value

A matrix of predictions.
