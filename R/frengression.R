#' Create a Frengression Model
#'
#' Constructs a frengression model for cross-sectional causal data simulation.
#' The model learns the joint distribution P(X, Y, Z) decomposed into three
#' sub-networks: the marginal P(X,Z), the causal margin P(Y|do(X)), and the
#' association remainder eta(X,Z).
#'
#' @param x_dim Dimension of treatment variable X.
#' @param y_dim Dimension of outcome variable Y.
#' @param z_dim Dimension of confounder variable Z.
#' @param num_layer Number of layers in each sub-network (default: 3).
#' @param hidden_dim Hidden layer width (default: 100).
#' @param noise_dim Noise dimension for stochastic generation (default: 10).
#' @param x_binary Logical; is X binary? (default: FALSE).
#' @param z_binary_dims Number of leading Z dimensions that are binary (default: 0).
#' @param y_binary Logical; is Y binary? (default: FALSE).
#'
#' @return A frengression object (S3 class) containing three sub-networks
#'   and model parameters.
#'
#' @examples
#' \donttest{
#'   model <- frengression(x_dim = 1, y_dim = 1, z_dim = 1)
#' }
#'
#' @export
frengression <- function(x_dim, y_dim, z_dim,
                         num_layer = 3, hidden_dim = 100, noise_dim = 10,
                         x_binary = FALSE, z_binary_dims = 0, y_binary = FALSE) {

  out_act <- if (y_binary) "sigmoid" else NULL

  model_xz <- build_sto_net(0, x_dim + z_dim, num_layer, hidden_dim,
                            max(x_dim + z_dim, noise_dim))
  model_y <- build_sto_net(x_dim + y_dim, y_dim, num_layer, hidden_dim,
                           noise_dim, out_act = out_act)
  model_eta <- build_sto_net(x_dim + z_dim, y_dim, num_layer, hidden_dim,
                             noise_dim)

  obj <- list(
    model_xz = model_xz,
    model_y = model_y,
    model_eta = model_eta,
    x_dim = x_dim,
    y_dim = y_dim,
    z_dim = z_dim,
    num_layer = num_layer,
    hidden_dim = hidden_dim,
    noise_dim = noise_dim,
    x_binary = x_binary,
    z_binary_dims = z_binary_dims,
    y_binary = y_binary,
    lossvec_y = NULL,
    lossvec_xz = NULL,
    causal_fn = NULL  # user-specified causal margin (if any)
  )
  class(obj) <- "frengression"
  obj
}

#' Train the Outcome Model (Y)
#'
#' Jointly trains model_y (causal margin) and model_eta (association remainder).
#' Uses the Z-permutation trick to ensure eta becomes marginally N(0,I).
#'
#' @param model A frengression, frengression_seq, or frengression_surv object.
#' @param ... Additional arguments passed to methods.
#' @return The model with updated weights and training loss history.
#' @export
train_y <- function(model, ...) UseMethod("train_y")

#' @param x Treatment matrix or tensor (n x x_dim).
#' @param z Confounder matrix or tensor (n x z_dim).
#' @param y Outcome matrix or tensor (n x y_dim).
#' @param num_iters Number of training iterations (default: 100).
#' @param lr Learning rate (default: 1e-3).
#' @param print_every Number of iterations between progress prints (default: 10).
#' @param tol Early stopping tolerance (default: 0.01).
#' @param silent Suppress output (default: FALSE).
#' @rdname train_y
#' @export
train_y.frengression <- function(model, x, z, y,
                                  num_iters = 100, lr = 1e-3,
                                  print_every = 10, tol = 0.01,
                                  silent = FALSE, ...) {
  x <- as_tensor(x)
  z <- as_tensor(z)
  y <- as_tensor(y)
  n <- x$size(1)

  model$model_y$model$train()
  model$model_eta$model$train()

  all_params <- c(model$model_y$model$parameters, model$model_eta$model$parameters)
  optimizer <- optim_adam(all_params, lr = lr)

  xz <- torch_cat(list(x, z), dim = 2)
  lossvec <- matrix(nrow = num_iters, ncol = 4)
  colnames(lossvec) <- c("loss", "loss_y", "loss1_y", "loss_eta")
  n_completed <- 0

  for (i in seq_len(num_iters)) {
    optimizer$zero_grad()

    # Forward: eta from (x, z), then y from (x, eta)
    eta1 <- model$model_eta$forward(xz)
    eta2 <- model$model_eta$forward(xz)
    y_sample1 <- model$model_y$forward(torch_cat(list(x, eta1), dim = 2))
    y_sample2 <- model$model_y$forward(torch_cat(list(x, eta2), dim = 2))

    el_y <- energy_loss(y, y_sample1, y_sample2)

    # Z-permutation: decouple Z from X
    eta_true <- torch_randn(c(n, model$y_dim))
    z_perm1 <- z[torch_randperm(n) + 1L, ]
    z_perm2 <- z[torch_randperm(n) + 1L, ]
    xz_decouple1 <- torch_cat(list(x, z_perm1), dim = 2)
    xz_decouple2 <- torch_cat(list(x, z_perm2), dim = 2)
    eta1_d <- model$model_eta$forward(xz_decouple1)
    eta2_d <- model$model_eta$forward(xz_decouple2)

    el_eta <- energy_loss(eta_true, eta1_d, eta2_d)

    # Early stopping
    if (abs(as.numeric(el_y$loss2) - as.numeric(el_y$loss1)) < tol &&
        abs(as.numeric(el_eta$loss2) - as.numeric(el_eta$loss1)) < tol) {
      if (!silent) cat("Stopping at iter", i, "\n")
      break
    }

    loss <- el_y$loss + el_eta$loss
    loss$backward()
    optimizer$step()

    lossvec[i, ] <- c(as.numeric(loss), as.numeric(el_y$loss),
                      as.numeric(el_y$loss1), as.numeric(el_eta$loss))
    n_completed <- n_completed + 1

    if (!silent && ((i == 1) || (i %% print_every == 0))) {
      cat(sprintf("Epoch %d: loss %.4f, loss_y %.4f (%.4f, %.4f), loss_eta %.4f (%.4f, %.4f)\n",
                  i, as.numeric(loss), as.numeric(el_y$loss),
                  as.numeric(el_y$loss1), as.numeric(el_y$loss2),
                  as.numeric(el_eta$loss), as.numeric(el_eta$loss1),
                  as.numeric(el_eta$loss2)))
    }
  }

  model$model_y$model$eval()
  model$model_eta$model$eval()
  model$lossvec_y <- lossvec[seq_len(n_completed), , drop = FALSE]
  model
}

#' Train the Marginal Model (XZ)
#'
#' Trains model_xz to generate the joint marginal distribution of
#' treatment X and confounders Z.
#'
#' @param model A frengression, frengression_seq, or frengression_surv object.
#' @param ... Additional arguments passed to methods.
#' @return The model with updated weights and training loss history.
#' @export
train_xz <- function(model, ...) UseMethod("train_xz")

#' @param x Treatment matrix or tensor (n x x_dim).
#' @param z Confounder matrix or tensor (n x z_dim).
#' @param num_iters Number of training iterations (default: 100).
#' @param lr Learning rate (default: 1e-4).
#' @param print_every Number of iterations between progress prints (default: 10).
#' @param silent Suppress output (default: FALSE).
#' @rdname train_xz
#' @export
train_xz.frengression <- function(model, x, z,
                                   num_iters = 100, lr = 1e-4,
                                   print_every = 10, silent = FALSE, ...) {
  x <- as_tensor(x)
  z <- as_tensor(z)
  n <- x$size(1)

  model$model_xz$model$train()
  optimizer <- optim_adam(model$model_xz$model$parameters, lr = lr)
  xz <- torch_cat(list(x, z), dim = 2)

  lossvec <- matrix(nrow = num_iters, ncol = 3)
  colnames(lossvec) <- c("loss", "loss1", "loss2")

  for (i in seq_len(num_iters)) {
    optimizer$zero_grad()

    sample1 <- model$model_xz$forward(n = n)
    sample2 <- model$model_xz$forward(n = n)

    # Apply sigmoid for binary variables
    if (model$x_binary) {
      sample1[, 1:model$x_dim] <- torch_sigmoid(sample1[, 1:model$x_dim])
      sample2[, 1:model$x_dim] <- torch_sigmoid(sample2[, 1:model$x_dim])
    }
    if (model$z_binary_dims > 0) {
      idx_start <- model$x_dim + 1L
      idx_end <- model$x_dim + model$z_binary_dims
      sample1[, idx_start:idx_end] <- torch_sigmoid(sample1[, idx_start:idx_end])
      sample2[, idx_start:idx_end] <- torch_sigmoid(sample2[, idx_start:idx_end])
    }

    el <- energy_loss(xz, sample1, sample2)
    el$loss$backward()
    optimizer$step()

    lossvec[i, ] <- c(as.numeric(el$loss), as.numeric(el$loss1), as.numeric(el$loss2))

    if (!silent && ((i == 1) || (i %% print_every == 0))) {
      cat(sprintf("Epoch %d: loss %.4f, loss1 %.4f, loss2 %.4f\n",
                  i, as.numeric(el$loss), as.numeric(el$loss1), as.numeric(el$loss2)))
    }
  }

  model$model_xz$model$eval()
  model$lossvec_xz <- lossvec
  model
}
