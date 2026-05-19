# Reset Outcome Models

Reinitializes model_y and model_eta with fresh random weights. Also
clears any user-specified causal margin.

## Usage

``` r
reset_y_models(model, ...)

# S3 method for class 'frengression'
reset_y_models(model, ...)
```

## Arguments

- model:

  A frengression object.

- ...:

  Additional arguments (ignored).

## Value

The model with reinitialized outcome networks.
