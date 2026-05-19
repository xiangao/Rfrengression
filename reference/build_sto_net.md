# Build a Stochastic Network

Creates an nn_sequential model where the first layer accepts in_dim +
noise_dim inputs. Noise is concatenated at the input layer before each
forward pass, following the engression architecture.

## Usage

``` r
build_sto_net(
  in_dim,
  out_dim,
  num_layer = 3,
  hidden_dim = 100,
  noise_dim = 10,
  out_act = NULL
)
```

## Arguments

- in_dim:

  Input dimension (0 for unconditional generators).

- out_dim:

  Output dimension.

- num_layer:

  Number of layers (minimum 2).

- hidden_dim:

  Hidden layer width.

- noise_dim:

  Dimension of noise concatenated at input.

- out_act:

  Output activation: NULL (identity) or "sigmoid".

## Value

A list with components: model (nn_sequential), noise_dim, in_dim,
out_dim, and forward (a function that takes input x or sample_size n and
returns output).
