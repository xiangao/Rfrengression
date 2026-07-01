#' Create a FrengressionSeq Model for Longitudinal Data
#'
#' Constructs a frengression model for time-varying/longitudinal causal data.
#' Maintains lists of T networks for each component, with sequential
#' dependence across time steps.
#'
#' @param x_dim Dimension of treatment at each time step.
#' @param y_dim Dimension of outcome at each time step.
#' @param z_dim Dimension of time-varying confounder at each time step.
#' @param T_steps Number of time steps.
#' @param s_dim Dimension of static baseline variables.
#' @param num_layer Number of layers (default: 3).
#' @param hidden_dim Hidden layer width (default: 100).
#' @param noise_dim Noise dimension (default: 10).
#' @param x_binary Logical; is X binary? (default: FALSE).
#' @param z_binary Logical; is Z binary? (default: FALSE).
#' @param y_binary Logical; is Y binary? (default: FALSE).
#' @param s_binary_dims Number of binary baseline dimensions (default: 0).
#' @param s_in_predict Include S in prediction models (default: TRUE).
#'
#' @return A frengression_seq object.
#' @export
frengression_seq <- function(x_dim, y_dim, z_dim, T_steps, s_dim,
                             num_layer = 3, hidden_dim = 100, noise_dim = 10,
                             x_binary = FALSE, z_binary = FALSE, y_binary = FALSE,
                             s_binary_dims = 0, s_in_predict = TRUE) {

  out_act <- if (y_binary) "sigmoid" else NULL

  # model_xz: list of T networks
  # t=1: S -> (X0, Z0)
  model_xz <- list(
    build_sto_net(s_dim, x_dim + z_dim, num_layer, hidden_dim,
                  max(x_dim + z_dim, noise_dim))
  )
  for (t in seq_len(T_steps - 1)) {
    model_xz[[t + 1]] <- build_sto_net(
      s_dim + (x_dim + z_dim) * t, x_dim + z_dim,
      num_layer, hidden_dim, max(x_dim + z_dim + y_dim, noise_dim)
    )
  }

  # model_y: list of T networks
  if (s_in_predict) {
    model_y <- list(
      build_sto_net(s_dim + x_dim + y_dim, y_dim, num_layer, hidden_dim,
                    noise_dim, out_act = out_act)
    )
    for (t in seq(2, T_steps)) {
      model_y[[t]] <- build_sto_net(
        s_dim + x_dim * t + y_dim, y_dim, num_layer, hidden_dim,
        noise_dim, out_act = out_act
      )
    }
  } else {
    model_y <- list(
      build_sto_net(x_dim + y_dim, y_dim, num_layer, hidden_dim,
                    noise_dim, out_act = out_act)
    )
    for (t in seq(2, T_steps)) {
      model_y[[t]] <- build_sto_net(
        x_dim * t + y_dim, y_dim, num_layer, hidden_dim,
        noise_dim, out_act = out_act
      )
    }
  }

  # model_e: list of T networks (confounder model)
  if (s_in_predict) {
    model_e <- list(
      build_sto_net(s_dim, z_dim, num_layer, hidden_dim, noise_dim)
    )
    for (t in seq(2, T_steps)) {
      model_e[[t]] <- build_sto_net(
        s_dim + x_dim * (t - 1), z_dim, num_layer, hidden_dim, noise_dim
      )
    }
  } else {
    model_e <- list(
      build_sto_net(x_dim, z_dim, num_layer, hidden_dim, noise_dim)
    )
    for (t in seq(2, T_steps)) {
      model_e[[t]] <- build_sto_net(
        x_dim * t, z_dim, num_layer, hidden_dim, noise_dim
      )
    }
  }

  # model_eta: list of T networks
  model_eta <- list(
    build_sto_net(s_dim + x_dim + z_dim, y_dim, num_layer, hidden_dim, noise_dim)
  )
  for (t in seq(2, T_steps)) {
    model_eta[[t]] <- build_sto_net(
      s_dim + (x_dim + z_dim) * t, y_dim, num_layer, hidden_dim, noise_dim
    )
  }

  obj <- list(
    model_xz = model_xz, model_y = model_y,
    model_eta = model_eta, model_e = model_e,
    x_dim = x_dim, y_dim = y_dim, z_dim = z_dim,
    s_dim = s_dim, T_steps = T_steps,
    num_layer = num_layer, hidden_dim = hidden_dim, noise_dim = noise_dim,
    x_binary = x_binary, z_binary = z_binary, y_binary = y_binary,
    s_binary_dims = s_binary_dims, s_in_predict = s_in_predict
  )
  class(obj) <- "frengression_seq"
  obj
}

# --- Internal helpers for sequential sampling ---

#' @keywords internal
.seq_sample_xz <- function(model, s, x = NULL, z = NULL) {
  xz <- model$model_xz[[1]]$forward(s)
  x0 <- xz[, 1:model$x_dim, drop = FALSE]
  z0 <- xz[, (model$x_dim + 1):(model$x_dim + model$z_dim), drop = FALSE]
  if (model$x_binary) x0 <- (x0 > 0)$to(dtype = torch_float())
  if (model$z_binary) z0 <- (z0 > 0)$to(dtype = torch_float())

  x_all <- list(x0)
  z_all <- list(z0)

  for (t in seq(2, model$T_steps)) {
    if (!is.null(x)) {
      sxz_p <- torch_cat(list(s,
                               x[, 1:((t - 1) * model$x_dim), drop = FALSE],
                               z[, 1:((t - 1) * model$z_dim), drop = FALSE]), dim = 2)
    } else {
      # Must match the cumulative-history convention used by the training
      # path above (and by .seq_sample_eta/.seq_sample_y): concatenate ALL
      # generated (x, z) up to t-1, not just the single previous time step.
      # model_xz[[t]] was built expecting that growing width; feeding only
      # the last step's slice previously crashed with a shape mismatch for
      # any T_steps >= 3 (T_steps == 2 happens to coincide, which is why
      # this went unnoticed).
      sxz_p <- torch_cat(c(list(s), x_all[seq_len(t - 1)], z_all[seq_len(t - 1)]), dim = 2)
    }
    xz <- model$model_xz[[t]]$forward(sxz_p)
    xt <- xz[, 1:model$x_dim, drop = FALSE]
    zt <- xz[, (model$x_dim + 1):(model$x_dim + model$z_dim), drop = FALSE]
    if (model$x_binary) xt <- (xt > 0)$to(dtype = torch_float())
    if (model$z_binary) zt <- (zt > 0)$to(dtype = torch_float())
    x_all[[t]] <- xt
    z_all[[t]] <- zt
  }
  list(x = torch_cat(x_all, dim = 2), z = torch_cat(z_all, dim = 2))
}

#' @keywords internal
.seq_sample_eta <- function(model, s, x, z) {
  eta_all <- list()
  for (t in seq_len(model$T_steps)) {
    sxz_p <- torch_cat(list(s,
                             x[, 1:(t * model$x_dim), drop = FALSE],
                             z[, 1:(t * model$z_dim), drop = FALSE]), dim = 2)
    eta_all[[t]] <- model$model_eta[[t]]$forward(sxz_p)
  }
  torch_cat(eta_all, dim = 2)
}

#' @keywords internal
.seq_sample_e <- function(model, s, x) {
  e_all <- list(model$model_e[[1]]$forward(s))
  for (t in seq(2, model$T_steps)) {
    e_all[[t]] <- model$model_e[[t]]$forward(
      torch_cat(list(s, x[, 1:((t - 1) * model$x_dim), drop = FALSE]), dim = 2)
    )
  }
  torch_cat(e_all, dim = 2)
}

#' @keywords internal
.seq_sample_y <- function(model, s, x, eta) {
  y_all <- list()
  for (t in seq_len(model$T_steps)) {
    if (model$s_in_predict) {
      inp <- torch_cat(list(s,
                             x[, 1:(t * model$x_dim), drop = FALSE],
                             eta[, ((t - 1) * model$y_dim + 1):(t * model$y_dim), drop = FALSE]),
                        dim = 2)
    } else {
      inp <- torch_cat(list(x[, 1:(t * model$x_dim), drop = FALSE],
                             eta[, ((t - 1) * model$y_dim + 1):(t * model$y_dim), drop = FALSE]),
                        dim = 2)
    }
    y_all[[t]] <- model$model_y[[t]]$forward(inp)
  }
  torch_cat(y_all, dim = 2)
}

# --- Training methods for FrengressionSeq ---

#' @param s Static baseline variables (matrix or tensor).
#' @param x Treatment matrix or tensor.
#' @param z Confounder matrix or tensor.
#' @param num_iters Number of training iterations (default: 100).
#' @param lr Learning rate (default: 1e-3).
#' @param print_every Number of iterations between progress prints (default: 10).
#' @param silent Suppress output (default: FALSE).
#' @rdname train_xz
#' @export
train_xz.frengression_seq <- function(model, s, x, z,
                                       num_iters = 100, lr = 1e-3,
                                       print_every = 10, silent = FALSE, ...) {
  s <- as_tensor(s)
  x <- as_tensor(x)
  z <- as_tensor(z)

  all_params <- list()
  for (t in seq_len(model$T_steps)) {
    model$model_xz[[t]]$model$train()
    all_params <- c(all_params, model$model_xz[[t]]$model$parameters)
  }
  optimizer <- optim_adam(all_params, lr = lr)

  xz <- torch_cat(list(x, z), dim = 2)

  for (i in seq_len(num_iters)) {
    optimizer$zero_grad()
    res1 <- .seq_sample_xz(model, s, x, z)
    sample1 <- torch_cat(list(res1$x, res1$z), dim = 2)
    res2 <- .seq_sample_xz(model, s, x, z)
    sample2 <- torch_cat(list(res2$x, res2$z), dim = 2)

    if (model$x_binary) {
      sample1[, 1:model$x_dim] <- torch_sigmoid(sample1[, 1:model$x_dim])
      sample2[, 1:model$x_dim] <- torch_sigmoid(sample2[, 1:model$x_dim])
    }
    if (model$z_binary) {
      zstart <- model$x_dim * model$T_steps + 1L
      sample1[, zstart:ncol(sample1)] <- torch_sigmoid(sample1[, zstart:ncol(sample1)])
      sample2[, zstart:ncol(sample2)] <- torch_sigmoid(sample2[, zstart:ncol(sample2)])
    }

    el <- energy_loss(xz, sample1, sample2)
    el$loss$backward()
    optimizer$step()

    if (!silent && ((i == 1) || (i %% print_every == 0))) {
      cat(sprintf("Epoch %d: loss %.4f, loss1 %.4f, loss2 %.4f\n",
                  i, as.numeric(el$loss), as.numeric(el$loss1), as.numeric(el$loss2)))
    }
  }

  for (t in seq_len(model$T_steps)) model$model_xz[[t]]$model$eval()
  model
}

#' Train the Confounder Model (E)
#'
#' Trains model_e to learn P(Z | pa(X)) for the sequential model.
#'
#' @param model A frengression_seq or frengression_surv object.
#' @param ... Additional arguments passed to methods.
#' @return The model with updated weights.
#' @export
train_e <- function(model, ...) UseMethod("train_e")

#' @param s Static baseline variables (matrix or tensor).
#' @param x Treatment matrix or tensor.
#' @param z Confounder matrix or tensor.
#' @param num_iters Number of training iterations (default: 100).
#' @param lr Learning rate (default: 1e-3).
#' @param print_every Number of iterations between progress prints (default: 10).
#' @param silent Suppress output (default: FALSE).
#' @rdname train_e
#' @export
train_e.frengression_seq <- function(model, s, x, z,
                                      num_iters = 100, lr = 1e-3,
                                      print_every = 10, silent = FALSE, ...) {
  s <- as_tensor(s)
  x <- as_tensor(x)
  z <- as_tensor(z)

  all_params <- list()
  for (t in seq_len(model$T_steps)) {
    model$model_e[[t]]$model$train()
    all_params <- c(all_params, model$model_e[[t]]$model$parameters)
  }
  optimizer <- optim_adam(all_params, lr = lr)

  for (i in seq_len(num_iters)) {
    optimizer$zero_grad()
    sample1 <- .seq_sample_e(model, s, x)
    sample2 <- .seq_sample_e(model, s, x)

    el <- energy_loss(z, sample1, sample2)
    el$loss$backward()
    optimizer$step()

    if (!silent && ((i == 1) || (i %% print_every == 0))) {
      cat(sprintf("Epoch %d: loss %.4f, loss1 %.4f, loss2 %.4f\n",
                  i, as.numeric(el$loss), as.numeric(el$loss1), as.numeric(el$loss2)))
    }
  }

  for (t in seq_len(model$T_steps)) model$model_e[[t]]$model$eval()
  model
}

#' @param s Static baseline variables (matrix or tensor).
#' @param x Treatment matrix or tensor.
#' @param z Confounder matrix or tensor.
#' @param y Outcome matrix or tensor.
#' @param num_iters Number of training iterations (default: 100).
#' @param lr Learning rate (default: 1e-4).
#' @param print_every Number of iterations between progress prints (default: 10).
#' @param silent Suppress output (default: FALSE).
#' @rdname train_y
#' @export
train_y.frengression_seq <- function(model, s, x, z, y,
                                      num_iters = 100, lr = 1e-4,
                                      print_every = 10, silent = FALSE, ...) {
  s <- as_tensor(s)
  x <- as_tensor(x)
  z <- as_tensor(z)
  y <- as_tensor(y)
  n <- x$size(1)

  all_params <- list()
  for (t in seq_len(model$T_steps)) {
    model$model_y[[t]]$model$train()
    model$model_eta[[t]]$model$train()
    all_params <- c(all_params, model$model_y[[t]]$model$parameters,
                    model$model_eta[[t]]$model$parameters)
  }
  optimizer <- optim_adam(all_params, lr = lr)

  for (i in seq_len(num_iters)) {
    optimizer$zero_grad()

    eta1 <- .seq_sample_eta(model, s, x, z)
    eta2 <- .seq_sample_eta(model, s, x, z)
    y_sample1 <- .seq_sample_y(model, s, x, eta1)
    y_sample2 <- .seq_sample_y(model, s, x, eta2)

    el_y <- energy_loss(y, y_sample1, y_sample2)

    # Marginal constraint via model_e
    eta_true <- torch_randn(c(n, model$y_dim * model$T_steps))
    z1 <- .seq_sample_e(model, s, x)
    z2 <- .seq_sample_e(model, s, x)
    eta1_d <- .seq_sample_eta(model, s, x, z1)
    eta2_d <- .seq_sample_eta(model, s, x, z2)

    el_eta <- energy_loss(eta_true, eta1_d, eta2_d)

    loss <- el_y$loss + el_eta$loss
    loss$backward()
    optimizer$step()

    if (!silent && ((i == 1) || (i %% print_every == 0))) {
      cat(sprintf("Epoch %d: loss %.4f, loss_y %.4f (%.4f, %.4f), loss_eta %.4f (%.4f, %.4f)\n",
                  i, as.numeric(loss), as.numeric(el_y$loss),
                  as.numeric(el_y$loss1), as.numeric(el_y$loss2),
                  as.numeric(el_eta$loss), as.numeric(el_eta$loss1),
                  as.numeric(el_eta$loss2)))
    }
  }

  for (t in seq_len(model$T_steps)) {
    model$model_y[[t]]$model$eval()
    model$model_eta[[t]]$model$eval()
  }
  model
}

# --- Sampling methods for FrengressionSeq ---

#' @param s Static baseline variables (matrix or tensor).
#' @rdname sample_joint
#' @export
sample_joint.frengression_seq <- function(model, s, ...) {
  s <- as_tensor(s)
  with_no_grad({
    res_xz <- .seq_sample_xz(model, s)
    x_all <- res_xz$x
    z_all <- res_xz$z
    eta_all <- .seq_sample_eta(model, s, x_all, z_all)
    y_all <- .seq_sample_y(model, s, x_all, eta_all)

    if (model$y_binary) {
      y_all <- (y_all > 0.5)$to(dtype = torch_int32())
    }

    list(x = as.matrix(x_all), z = as.matrix(z_all), y = as.matrix(y_all))
  })
}

#' @param s Static baseline variables (matrix or tensor).
#' @rdname sample_xz
#' @export
sample_xz.frengression_seq <- function(model, s, ...) {
  s <- as_tensor(s)
  with_no_grad({
    res <- .seq_sample_xz(model, s)
    list(x = as.matrix(res$x), z = as.matrix(res$z))
  })
}

#' @param s Static baseline variables (matrix or tensor).
#' @param x Treatment matrix or tensor.
#' @param sample_size Number of samples (default: 100).
#' @rdname sample_causal_margin
#' @export
sample_causal_margin.frengression_seq <- function(model, s, x,
                                                   sample_size = 100, ...) {
  s <- as_tensor(s)
  x <- as_tensor(x)
  batch_size <- x$size(1)
  with_no_grad({
    all_y <- list()
    for (t in seq_len(model$T_steps)) {
      samples_t <- list()
      for (ss in seq_len(sample_size)) {
        eta <- torch_randn(c(batch_size, model$y_dim))
        if (model$s_in_predict) {
          inp <- torch_cat(list(s, x[, 1:(t * model$x_dim), drop = FALSE], eta), dim = 2)
        } else {
          inp <- torch_cat(list(x[, 1:(t * model$x_dim), drop = FALSE], eta), dim = 2)
        }
        yt <- model$model_y[[t]]$forward(inp)
        if (model$y_binary) yt <- (yt > 0.5)$to(dtype = torch_float())
        samples_t[[ss]] <- as.matrix(yt)
      }
      all_y[[t]] <- simplify2array(samples_t)
    }
    all_y
  })
}

#' @param s Static baseline variables (matrix or tensor).
#' @param x Treatment matrix or tensor.
#' @param target "mean" or "sample" (default: "mean").
#' @param sample_size Number of Monte Carlo samples (default: 100).
#' @param ... Additional arguments (ignored).
#' @rdname predict_causal
#' @export
predict_causal.frengression_seq <- function(model, s, x, target = "mean",
                                             sample_size = 100, ...) {
  s <- as_tensor(s)
  x <- as_tensor(x)
  batch_size <- x$size(1)
  with_no_grad({
    all_y <- list()
    for (t in seq_len(model$T_steps)) {
      samples_t <- list()
      for (ss in seq_len(sample_size)) {
        eta <- torch_randn(c(batch_size, model$y_dim))
        if (model$s_in_predict) {
          inp <- torch_cat(list(s, x[, 1:(t * model$x_dim), drop = FALSE], eta), dim = 2)
        } else {
          inp <- torch_cat(list(x[, 1:(t * model$x_dim), drop = FALSE], eta), dim = 2)
        }
        samples_t[[ss]] <- as.matrix(model$model_y[[t]]$forward(inp))
      }
      yt <- Reduce("+", samples_t) / sample_size
      if (model$y_binary) yt <- (yt > 0.5) * 1L
      all_y[[t]] <- yt
    }
    all_y
  })
}

#' @param s Static baseline variables (matrix or tensor).
#' @rdname predict.frengression
#' @export
predict.frengression_seq <- function(object, s, x, type = "mean",
                                      nsample = 200, ...) {
  predict_causal.frengression_seq(object, s, x, target = type,
                                   sample_size = nsample)
}
