#' Sample from the Joint Distribution
#'
#' Generates samples from the learned joint distribution P(X, Y, Z).
#'
#' @param model A frengression, frengression_seq, or frengression_surv object.
#' @param ... Additional arguments passed to methods.
#' @return A list with components x, y, z (matrices).
#' @export
sample_joint <- function(model, ...) UseMethod("sample_joint")

#' @param sample_size Number of samples to generate (default: 100).
#' @rdname sample_joint
#' @export
sample_joint.frengression <- function(model, sample_size = 100, ...) {
  with_no_grad({
    xz <- model$model_xz$forward(n = sample_size)
    x <- xz[, 1:model$x_dim, drop = FALSE]
    z <- xz[, (model$x_dim + 1):(model$x_dim + model$z_dim), drop = FALSE]

    if (model$x_binary) x <- (x > 0)$to(dtype = torch_float())
    if (model$z_binary_dims > 0) {
      z[, 1:model$z_binary_dims] <- (z[, 1:model$z_binary_dims] > 0)$to(dtype = torch_float())
    }

    xz_cat <- torch_cat(list(x, z), dim = 2)
    eta <- model$model_eta$forward(xz_cat)

    if (is.null(model$causal_fn)) {
      y <- model$model_y$forward(torch_cat(list(x, eta), dim = 2))
    } else {
      y <- model$causal_fn(x, eta)
    }

    if (model$y_binary) y <- (y > 0.5)$to(dtype = torch_float())

    list(x = as.matrix(x), y = as.matrix(y), z = as.matrix(z))
  })
}

#' Sample from the Marginal Distribution of (X, Z)
#'
#' Generates samples from the learned marginal distribution P(X, Z).
#'
#' @param model A frengression, frengression_seq, or frengression_surv object.
#' @param ... Additional arguments passed to methods.
#' @return A list with components x, z (matrices).
#' @export
sample_xz <- function(model, ...) UseMethod("sample_xz")

#' @param sample_size Number of samples to generate (default: 100).
#' @rdname sample_xz
#' @export
sample_xz.frengression <- function(model, sample_size = 100, ...) {
  with_no_grad({
    xz <- model$model_xz$forward(n = sample_size)
    x <- xz[, 1:model$x_dim, drop = FALSE]
    z <- xz[, (model$x_dim + 1):(model$x_dim + model$z_dim), drop = FALSE]

    if (model$x_binary) x <- (x > 0)$to(dtype = torch_float())
    if (model$z_binary_dims > 0) {
      z[, 1:model$z_binary_dims] <- (z[, 1:model$z_binary_dims] > 0)$to(dtype = torch_float())
    }

    list(x = as.matrix(x), z = as.matrix(z))
  })
}

#' Sample from the Causal Margin P(Y | do(X))
#'
#' Generates outcome samples under an intervention do(X = x),
#' marginalizing over the noise distribution.
#'
#' @param model A frengression, frengression_seq, or frengression_surv object.
#' @param ... Additional arguments passed to methods.
#' @return A matrix of outcome samples (sample_size x y_dim).
#' @export
sample_causal_margin <- function(model, ...) UseMethod("sample_causal_margin")

#' @param x Intervention values for X (matrix or vector).
#' @param sample_size Number of samples per intervention value (default: 100).
#' @rdname sample_causal_margin
#' @export
sample_causal_margin.frengression <- function(model, x, sample_size = 100, ...) {
  x <- as_tensor(x)
  batch_size <- x$size(1)
  with_no_grad({
    samples <- list()
    for (s in seq_len(sample_size)) {
      eta <- torch_randn(c(batch_size, model$y_dim))
      if (is.null(model$causal_fn)) {
        x_eta <- torch_cat(list(x, eta), dim = 2)
        ys <- model$model_y$forward(x_eta)
      } else {
        ys <- model$causal_fn(x, eta)
      }
      if (model$y_binary) {
        ys <- (ys > 0.5)$to(dtype = torch_float())
      }
      samples[[s]] <- as.matrix(ys)
    }
    simplify2array(samples)
  })
}
