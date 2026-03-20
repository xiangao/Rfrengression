#' Build a Stochastic Network
#'
#' Creates an nn_sequential model where the first layer accepts
#' in_dim + noise_dim inputs. Noise is concatenated at the input layer
#' before each forward pass, following the engression architecture.
#'
#' @param in_dim Input dimension (0 for unconditional generators).
#' @param out_dim Output dimension.
#' @param num_layer Number of layers (minimum 2).
#' @param hidden_dim Hidden layer width.
#' @param noise_dim Dimension of noise concatenated at input.
#' @param out_act Output activation: NULL (identity) or "sigmoid".
#' @return A list with components: model (nn_sequential), noise_dim, in_dim,
#'   out_dim, and forward (a function that takes input x or sample_size n
#'   and returns output).
#' @keywords internal
build_sto_net <- function(in_dim, out_dim, num_layer = 3, hidden_dim = 100,
                          noise_dim = 10, out_act = NULL) {
  layers <- list()

  # First layer: in_dim + noise_dim -> hidden_dim
  layers <- c(layers, list(
    nn_linear(in_dim + noise_dim, hidden_dim),
    nn_elu()
  ))

  # Hidden layers
  if (num_layer > 2) {
    for (lay in seq_len(num_layer - 2)) {
      layers <- c(layers, list(
        nn_linear(hidden_dim, hidden_dim),
        nn_elu()
      ))
    }
  }

  # Output layer
  layers <- c(layers, list(nn_linear(hidden_dim, out_dim)))

  # Output activation
  if (!is.null(out_act) && out_act == "sigmoid") {
    layers <- c(layers, list(nn_sigmoid()))
  }

  model <- do.call(nn_sequential, layers)

  # Forward function: concatenates noise before passing through network
  forward_fn <- function(x = NULL, n = NULL) {
    if (is.null(x) && is.null(n)) stop("Provide either x (input tensor) or n (sample size)")
    if (is.null(x)) {
      # Unconditional: noise only
      noise <- torch_randn(n, noise_dim)
      return(model(noise))
    }
    if (!inherits(x, "torch_tensor")) x <- torch_tensor(x, dtype = torch_float())
    batch_size <- x$size(1)
    noise <- torch_randn(batch_size, noise_dim)
    if (in_dim == 0) {
      model(noise)
    } else {
      model(torch_cat(list(x, noise), dim = 2))
    }
  }

  # Sample function: generates multiple samples for same input
  sample_fn <- function(x, sample_size = 100) {
    if (!inherits(x, "torch_tensor")) x <- torch_tensor(x, dtype = torch_float())
    batch_size <- x$size(1)
    samples <- list()
    for (s in seq_len(sample_size)) {
      samples[[s]] <- forward_fn(x = x)
    }
    torch_stack(samples, dim = 3)  # batch x out_dim x sample_size
  }

  # Predict function: returns mean over samples
  predict_fn <- function(x, target = "mean", sample_size = 100) {
    samp <- sample_fn(x, sample_size)  # batch x out_dim x sample_size
    if (target == "mean") {
      as.matrix(torch_mean(samp, dim = 3))
    } else if (target == "sample") {
      samp
    } else {
      as.matrix(torch_mean(samp, dim = 3))
    }
  }

  list(
    model = model,
    noise_dim = noise_dim,
    in_dim = in_dim,
    out_dim = out_dim,
    forward = forward_fn,
    sample = sample_fn,
    predict = predict_fn
  )
}
