# Predict Causal Effects

Computes predictions from the causal margin P(Y\|do(X=x)), returning the
mean, quantiles, or raw samples. For the learned model, this
marginalizes over the noise input eta ~ N(0,I) that substitutes for the
confounders.

## Usage

``` r
# S3 method for class 'frengression_seq'
predict(object, s, x, type = "mean", nsample = 200, ...)

# S3 method for class 'frengression_surv'
predict(object, s, x, type = "mean", nsample = 200, ...)

# S3 method for class 'frengression'
predict(
  object,
  x,
  type = c("mean", "sample", "quantile"),
  nsample = 200,
  quantiles = seq(0.1, 0.9, 0.1),
  trim = 0.05,
  ...
)
```

## Arguments

- object:

  A frengression object.

- s:

  Static baseline variables (matrix or tensor).

- x:

  Intervention values for X (matrix, vector, or data frame).

- type:

  Type of prediction: "mean", "quantile", or "sample" (default: "mean").

- nsample:

  Number of Monte Carlo samples (default: 200).

- ...:

  Additional arguments (ignored).

- quantiles:

  Quantile levels if type = "quantile" (default: seq(0.1, 0.9, 0.1)).

- trim:

  Trimming proportion for mean (default: 0.05).

## Value

A matrix of predictions.
