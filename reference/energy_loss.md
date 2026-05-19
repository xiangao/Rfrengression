# Two-Sample Energy Loss

Computes the energy loss between observed values and two independent
stochastic predictions. The loss is E\|Y-Yhat\| - 0.5\*E\|Yhat-Yhat'\|.

## Usage

``` r
energy_loss(yt, mxt, mxpt)
```

## Arguments

- yt:

  Tensor of observed values.

- mxt:

  Tensor of first stochastic predictions.

- mxpt:

  Tensor of second stochastic predictions.

## Value

A list with components: loss (total), loss1 (prediction error), loss2
(diversity term).
