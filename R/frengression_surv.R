#' Create a FrengressionSurv Model for Survival Data
#'
#' Constructs a frengression model for survival/time-to-event data.
#' Similar to FrengressionSeq but with binary outcomes (events) and
#' censoring handling: values after the first event are masked during
#' training.
#'
#' @param x_dim Dimension of treatment at each time step.
#' @param y_dim Dimension of outcome at each time step (default: 1).
#' @param z_dim Dimension of time-varying confounder at each time step.
#' @param T_steps Number of time steps.
#' @param s_dim Dimension of static baseline variables.
#' @param num_layer Number of layers (default: 3).
#' @param hidden_dim Hidden layer width (default: 100).
#' @param noise_dim Noise dimension (default: 10).
#' @param x_binary Logical; is X binary? (default: FALSE).
#' @param z_binary Logical; is Z binary? (default: FALSE).
#' @param y_binary Logical; is Y binary? (default: TRUE for survival).
#' @param s_in_predict Include S in prediction models (default: TRUE).
#'
#' @return A frengression_surv object.
#' @export
frengression_surv <- function(x_dim, y_dim = 1, z_dim, T_steps, s_dim,
                              num_layer = 3, hidden_dim = 100, noise_dim = 10,
                              x_binary = FALSE, z_binary = FALSE, y_binary = TRUE,
                              s_in_predict = TRUE) {

  out_act <- if (y_binary) "sigmoid" else NULL

  # model_xz
  model_xz <- list(
    build_sto_net(s_dim, x_dim + z_dim, num_layer, hidden_dim, noise_dim)
  )
  for (t in seq_len(T_steps - 1)) {
    model_xz[[t + 1]] <- build_sto_net(
      s_dim + (x_dim + z_dim) * t, x_dim + z_dim,
      num_layer, hidden_dim, max(x_dim + z_dim + y_dim, noise_dim)
    )
  }

  # model_e
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

  # model_y
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

  # model_eta
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
    s_in_predict = s_in_predict
  )
  class(obj) <- "frengression_surv"
  obj
}

# --- Censoring mask helper ---

#' @keywords internal
.surv_censoring_mask <- function(y, y_dim) {
  # y is n x (T*y_dim); events are > 0
  # After first event, mask subsequent time steps
  event_indicator <- (y > 0)$to(dtype = torch_float())
  cc <- torch_cumsum(event_indicator, dim = 2)
  c_shifted <- torch_zeros_like(cc)
  if (cc$size(2) > 1) {
    c_shifted[, 2:cc$size(2)] <- cc[, 1:(cc$size(2) - 1)]
  }
  mask <- (c_shifted > 0)
  y_masked <- y$clone()
  y_masked$masked_fill_(mask, -1)
  list(y_masked = y_masked, mask = mask)
}

#' @keywords internal
.surv_resample_valid <- function(data_list, y_masked, t, model) {

  # Resample invalid (censored) rows from valid rows at time t
  # data_list contains x, z, s, y as tensors
  valid_idx <- (y_masked[, t + 1] >= -0.5)$nonzero()
  if (valid_idx$size(1) == 0) valid_idx <- torch_arange(1, y_masked$size(1))
  valid_idx <- valid_idx[, 1]

  invalid_idx <- (y_masked[, t + 1] < -0.5)$nonzero()
  if (invalid_idx$size(1) == 0) return(data_list)
  invalid_idx <- invalid_idx[, 1]

  sampled_idx <- valid_idx[torch_randint(1, valid_idx$size(1), invalid_idx$size(1), dtype = torch_long()) + 1L]

  result <- list()
  for (nm in names(data_list)) {
    d <- data_list[[nm]]$clone()
    d[invalid_idx + 1L, ] <- data_list[[nm]][sampled_idx + 1L, ]
    result[[nm]] <- d
  }
  result
}

# --- Training methods for FrengressionSurv ---

#' @param s Static baseline variables (matrix or tensor).
#' @param x Treatment matrix or tensor.
#' @param z Confounder matrix or tensor.
#' @param y Outcome matrix or tensor (events).
#' @param num_iters Number of training iterations (default: 100).
#' @param lr Learning rate (default: 1e-3).
#' @param print_every Number of iterations between progress prints (default: 10).
#' @param silent Suppress output (default: FALSE).
#' @rdname train_xz
#' @export
train_xz.frengression_surv <- function(model, s, x, z, y,
                                        num_iters = 100, lr = 1e-3,
                                        print_every = 10, silent = FALSE, ...) {
  s <- as_tensor(s)
  x <- as_tensor(x)
  z <- as_tensor(z)
  y <- as_tensor(y)
  n <- x$size(1)

  all_params <- list()
  for (t in seq_len(model$T_steps)) {
    model$model_xz[[t]]$model$train()
    all_params <- c(all_params, model$model_xz[[t]]$model$parameters)
  }
  optimizer <- optim_adam(all_params, lr = lr)

  for (i in seq_len(num_iters)) {
    cm <- .surv_censoring_mask(y, model$y_dim)
    y_masked <- cm$y_masked

    # Build resampled data lists per time step
    x_label_list <- list(x[, 1:model$x_dim, drop = FALSE])
    z_label_list <- list(z[, 1:model$z_dim, drop = FALSE])
    s_list <- list(s)
    x_hist_list <- list(x[, 1:model$x_dim, drop = FALSE])
    z_hist_list <- list(z[, 1:model$z_dim, drop = FALSE])

    for (t in seq(2, model$T_steps)) {
      tidx <- t  # 1-indexed time step
      valid_idx <- which(as.numeric(y_masked[, tidx]) >= -0.5)
      invalid_idx <- which(as.numeric(y_masked[, tidx]) < -0.5)

      x_t_label <- x[, ((t - 1) * model$x_dim + 1):(t * model$x_dim), drop = FALSE]
      z_t_label <- z[, ((t - 1) * model$z_dim + 1):(t * model$z_dim), drop = FALSE]
      x_t_hist <- x[, 1:((t - 1) * model$x_dim), drop = FALSE]
      z_t_hist <- z[, 1:((t - 1) * model$z_dim), drop = FALSE]
      s_t <- s$clone()

      if (length(invalid_idx) > 0 && length(valid_idx) > 0) {
        sampled <- valid_idx[sample.int(length(valid_idx), length(invalid_idx), replace = TRUE)]
        x_t_label[invalid_idx, ] <- x_t_label[sampled, ]
        z_t_label[invalid_idx, ] <- z_t_label[sampled, ]
        x_t_hist[invalid_idx, ] <- x[sampled, 1:((t - 1) * model$x_dim), drop = FALSE]
        z_t_hist[invalid_idx, ] <- z[sampled, 1:((t - 1) * model$z_dim), drop = FALSE]
        s_t[invalid_idx, ] <- s[sampled, ]
      }

      x_label_list[[t]] <- x_t_label
      z_label_list[[t]] <- z_t_label
      x_hist_list[[t]] <- x_t_hist
      z_hist_list[[t]] <- z_t_hist
      s_list[[t]] <- s_t
    }

    xz_target <- torch_cat(c(x_label_list, z_label_list), dim = 2)

    optimizer$zero_grad()

    # Generate samples from model
    xz1 <- model$model_xz[[1]]$forward(s)
    xz2 <- model$model_xz[[1]]$forward(s)
    if (model$x_binary) {
      xz1[, 1:model$x_dim] <- torch_sigmoid(xz1[, 1:model$x_dim])
      xz2[, 1:model$x_dim] <- torch_sigmoid(xz2[, 1:model$x_dim])
    }
    if (model$z_binary) {
      xz1[, (model$x_dim + 1):(model$x_dim + model$z_dim)] <-
        torch_sigmoid(xz1[, (model$x_dim + 1):(model$x_dim + model$z_dim)])
      xz2[, (model$x_dim + 1):(model$x_dim + model$z_dim)] <-
        torch_sigmoid(xz2[, (model$x_dim + 1):(model$x_dim + model$z_dim)])
    }
    x_s1 <- list(xz1[, 1:model$x_dim, drop = FALSE])
    z_s1 <- list(xz1[, (model$x_dim + 1):(model$x_dim + model$z_dim), drop = FALSE])
    x_s2 <- list(xz2[, 1:model$x_dim, drop = FALSE])
    z_s2 <- list(xz2[, (model$x_dim + 1):(model$x_dim + model$z_dim), drop = FALSE])

    for (t in seq(2, model$T_steps)) {
      sxz_p <- torch_cat(list(s_list[[t]], x_hist_list[[t]], z_hist_list[[t]]), dim = 2)
      xz1 <- model$model_xz[[t]]$forward(sxz_p)
      xz2 <- model$model_xz[[t]]$forward(sxz_p)
      if (model$x_binary) {
        xz1[, 1:model$x_dim] <- torch_sigmoid(xz1[, 1:model$x_dim])
        xz2[, 1:model$x_dim] <- torch_sigmoid(xz2[, 1:model$x_dim])
      }
      if (model$z_binary) {
        xz1[, (model$x_dim + 1):(model$x_dim + model$z_dim)] <-
          torch_sigmoid(xz1[, (model$x_dim + 1):(model$x_dim + model$z_dim)])
        xz2[, (model$x_dim + 1):(model$x_dim + model$z_dim)] <-
          torch_sigmoid(xz2[, (model$x_dim + 1):(model$x_dim + model$z_dim)])
      }
      x_s1[[t]] <- xz1[, 1:model$x_dim, drop = FALSE]
      z_s1[[t]] <- xz1[, (model$x_dim + 1):(model$x_dim + model$z_dim), drop = FALSE]
      x_s2[[t]] <- xz2[, 1:model$x_dim, drop = FALSE]
      z_s2[[t]] <- xz2[, (model$x_dim + 1):(model$x_dim + model$z_dim), drop = FALSE]
    }

    sample1 <- torch_cat(c(x_s1, z_s1), dim = 2)
    sample2 <- torch_cat(c(x_s2, z_s2), dim = 2)

    el <- energy_loss(xz_target, sample1, sample2)
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

#' @param s Static baseline variables (matrix or tensor).
#' @param x Treatment matrix or tensor.
#' @param z Confounder matrix or tensor.
#' @param y Outcome matrix or tensor (events for censoring mask).
#' @param num_iters Number of training iterations (default: 100).
#' @param lr Learning rate (default: 1e-3).
#' @param print_every Number of iterations between progress prints (default: 10).
#' @param silent Suppress output (default: FALSE).
#' @rdname train_e
#' @export
train_e.frengression_surv <- function(model, s, x, z, y,
                                       num_iters = 100, lr = 1e-3,
                                       print_every = 10, silent = FALSE, ...) {
  s <- as_tensor(s)
  x <- as_tensor(x)
  z <- as_tensor(z)
  y <- as_tensor(y)
  n <- x$size(1)

  all_params <- list()
  for (t in seq_len(model$T_steps)) {
    model$model_e[[t]]$model$train()
    all_params <- c(all_params, model$model_e[[t]]$model$parameters)
  }
  optimizer <- optim_adam(all_params, lr = lr)

  for (i in seq_len(num_iters)) {
    cm <- .surv_censoring_mask(y, model$y_dim)
    y_masked <- cm$y_masked

    # Build resampled data
    z_label_list <- list(z[, 1:model$z_dim, drop = FALSE])
    s_list <- list(s)
    x_hist_list <- list(NULL)  # not used for t=1

    for (t in seq(2, model$T_steps)) {
      valid_idx <- which(as.numeric(y_masked[, t]) >= -0.5)
      invalid_idx <- which(as.numeric(y_masked[, t]) < -0.5)

      z_t_label <- z[, ((t - 1) * model$z_dim + 1):(t * model$z_dim), drop = FALSE]
      s_t <- s$clone()
      x_t_hist <- x[, 1:((t - 1) * model$x_dim), drop = FALSE]

      if (length(invalid_idx) > 0 && length(valid_idx) > 0) {
        sampled <- valid_idx[sample.int(length(valid_idx), length(invalid_idx), replace = TRUE)]
        z_t_label[invalid_idx, ] <- z[sampled, ((t - 1) * model$z_dim + 1):(t * model$z_dim), drop = FALSE]
        s_t[invalid_idx, ] <- s[sampled, ]
        x_t_hist[invalid_idx, ] <- x[sampled, 1:((t - 1) * model$x_dim), drop = FALSE]
      }

      z_label_list[[t]] <- z_t_label
      s_list[[t]] <- s_t
      x_hist_list[[t]] <- x_t_hist
    }

    z_target <- torch_cat(z_label_list, dim = 2)

    optimizer$zero_grad()

    z_s1 <- list(model$model_e[[1]]$forward(s))
    z_s2 <- list(model$model_e[[1]]$forward(s))
    if (model$z_binary) {
      z_s1[[1]] <- torch_sigmoid(z_s1[[1]])
      z_s2[[1]] <- torch_sigmoid(z_s2[[1]])
    }

    for (t in seq(2, model$T_steps)) {
      sx_p <- torch_cat(list(s_list[[t]], x_hist_list[[t]]), dim = 2)
      zp1 <- model$model_e[[t]]$forward(sx_p)
      zp2 <- model$model_e[[t]]$forward(sx_p)
      if (model$z_binary) {
        zp1 <- torch_sigmoid(zp1)
        zp2 <- torch_sigmoid(zp2)
      }
      z_s1[[t]] <- zp1
      z_s2[[t]] <- zp2
    }

    z_sample1 <- torch_cat(z_s1, dim = 2)
    z_sample2 <- torch_cat(z_s2, dim = 2)

    el <- energy_loss(z_target, z_sample1, z_sample2)
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
#' @param y Outcome matrix or tensor (events).
#' @param num_iters Number of training iterations (default: 100).
#' @param lr Learning rate (default: 1e-3).
#' @param print_every Number of iterations between progress prints (default: 10).
#' @param reg_lambda Regularization weight for marginal loss (default: 0).
#' @param silent Suppress output (default: FALSE).
#' @rdname train_y
#' @export
train_y.frengression_surv <- function(model, s, x, z, y,
                                       num_iters = 100, lr = 1e-3,
                                       print_every = 10, reg_lambda = 0,
                                       silent = FALSE, ...) {
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
    cm <- .surv_censoring_mask(y, model$y_dim)
    y_masked <- cm$y_masked

    # Build resampled data
    y_list <- list(y[, 1:model$y_dim, drop = FALSE])
    x_list <- list(x[, 1:model$x_dim, drop = FALSE])
    z_list <- list(z[, 1:model$z_dim, drop = FALSE])
    s_list <- list(s)

    for (t in seq(2, model$T_steps)) {
      valid_idx <- which(as.numeric(y_masked[, t]) >= -0.5)
      invalid_idx <- which(as.numeric(y_masked[, t]) < -0.5)

      y_t <- y[, ((t - 1) * model$y_dim + 1):(t * model$y_dim), drop = FALSE]
      x_t <- x[, 1:(t * model$x_dim), drop = FALSE]
      z_t <- z[, 1:(t * model$z_dim), drop = FALSE]
      s_t <- s$clone()

      if (length(invalid_idx) > 0 && length(valid_idx) > 0) {
        sampled <- valid_idx[sample.int(length(valid_idx), length(invalid_idx), replace = TRUE)]
        y_t[invalid_idx, ] <- y[sampled, ((t - 1) * model$y_dim + 1):(t * model$y_dim), drop = FALSE]
        x_t[invalid_idx, ] <- x[sampled, 1:(t * model$x_dim), drop = FALSE]
        z_t[invalid_idx, ] <- z[sampled, 1:(t * model$z_dim), drop = FALSE]
        s_t[invalid_idx, ] <- s[sampled, ]
      }

      y_list[[t]] <- y_t
      x_list[[t]] <- x_t
      z_list[[t]] <- z_t
      s_list[[t]] <- s_t
    }

    y_target <- torch_cat(y_list, dim = 2)

    optimizer$zero_grad()

    y_s1 <- list()
    y_s2 <- list()
    for (t in seq_len(model$T_steps)) {
      sxz_p <- torch_cat(list(s_list[[t]], x_list[[t]], z_list[[t]]), dim = 2)
      etat1 <- model$model_eta[[t]]$forward(sxz_p)
      etat2 <- model$model_eta[[t]]$forward(sxz_p)
      if (model$s_in_predict) {
        yt1 <- model$model_y[[t]]$forward(torch_cat(list(s_list[[t]], x_list[[t]], etat1), dim = 2))
        yt2 <- model$model_y[[t]]$forward(torch_cat(list(s_list[[t]], x_list[[t]], etat2), dim = 2))
      } else {
        yt1 <- model$model_y[[t]]$forward(torch_cat(list(x_list[[t]], etat1), dim = 2))
        yt2 <- model$model_y[[t]]$forward(torch_cat(list(x_list[[t]], etat2), dim = 2))
      }
      y_s1[[t]] <- yt1
      y_s2[[t]] <- yt2
    }

    y_sample1 <- torch_cat(y_s1, dim = 2)
    y_sample2 <- torch_cat(y_s2, dim = 2)
    el_y <- energy_loss(y_target, y_sample1, y_sample2)

    # Marginal constraint
    eta_true <- torch_randn(c(n, model$y_dim * model$T_steps))

    # Sample z from model_e
    z_s1 <- list(model$model_e[[1]]$forward(s_list[[1]]))
    z_s2 <- list(model$model_e[[1]]$forward(s_list[[1]]))
    for (t in seq(2, model$T_steps)) {
      z_s1[[t]] <- .surv_sample_e_partial(model, s_list[[t]],
                                           x_list[[t]][, 1:((t - 1) * model$x_dim), drop = FALSE], t - 1)
      z_s2[[t]] <- .surv_sample_e_partial(model, s_list[[t]],
                                           x_list[[t]][, 1:((t - 1) * model$x_dim), drop = FALSE], t - 1)
    }

    eta1 <- list()
    eta2 <- list()
    for (t in seq_len(model$T_steps)) {
      sxz_p1 <- torch_cat(list(s_list[[t]], x_list[[t]], z_s1[[t]]), dim = 2)
      sxz_p2 <- torch_cat(list(s_list[[t]], x_list[[t]], z_s2[[t]]), dim = 2)
      eta1[[t]] <- model$model_eta[[t]]$forward(sxz_p1)
      eta2[[t]] <- model$model_eta[[t]]$forward(sxz_p2)
    }
    eta1_cat <- torch_cat(eta1, dim = 2)
    eta2_cat <- torch_cat(eta2, dim = 2)

    el_eta <- energy_loss(eta_true, eta1_cat, eta2_cat)

    # Marginal loss regularizer
    marginal_loss <- torch_tensor(0)
    if (reg_lambda > 0) {
      mean_y <- y_target$to(dtype = torch_float())$mean(dim = 1)
      mean_ys1 <- y_sample1$to(dtype = torch_float())$mean(dim = 1)
      mean_ys2 <- y_sample2$to(dtype = torch_float())$mean(dim = 1)
      eps <- 1e-6
      marginal_loss <- ((mean_ys1 / (mean_y + eps) - 1)^2)$mean() +
        ((mean_ys2 / (mean_y + eps) - 1)^2)$mean()
    }

    loss <- el_y$loss + el_eta$loss + reg_lambda * marginal_loss
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

#' @keywords internal
.surv_sample_e_partial <- function(model, s, x, T_partial) {
  e_all <- list(model$model_e[[1]]$forward(s))
  if (T_partial > 1) {
    for (t in seq(2, T_partial)) {
      e_all[[t]] <- model$model_e[[t]]$forward(
        torch_cat(list(s, x[, 1:((t - 1) * model$x_dim), drop = FALSE]), dim = 2)
      )
    }
  }
  torch_cat(e_all, dim = 2)
}

# --- Sampling methods for FrengressionSurv ---

#' @param s Static baseline variables (matrix or tensor).
#' @rdname sample_joint
#' @export
sample_joint.frengression_surv <- function(model, s, ...) {
  s <- as_tensor(s)
  with_no_grad({
    # Sequential generation
    xz <- model$model_xz[[1]]$forward(s)
    x0 <- xz[, 1:model$x_dim, drop = FALSE]
    z0 <- xz[, (model$x_dim + 1):(model$x_dim + model$z_dim), drop = FALSE]
    if (model$x_binary) x0 <- (x0 > 0)$to(dtype = torch_float())
    if (model$z_binary) z0 <- (z0 > 0)$to(dtype = torch_float())

    x_all <- x0
    z_all <- z0

    sxz_p <- torch_cat(list(s, x0, z0), dim = 2)
    eta0 <- model$model_eta[[1]]$forward(sxz_p)
    sxeta0 <- torch_cat(list(s, x0, eta0), dim = 2)
    y0 <- (model$model_y[[1]]$forward(sxeta0) > 0.5)$to(dtype = torch_int32())
    y_all <- y0

    for (t in seq(2, model$T_steps)) {
      sxz_in <- torch_cat(list(s,
                                 x_all[, 1:((t - 1) * model$x_dim), drop = FALSE],
                                 z_all[, 1:((t - 1) * model$z_dim), drop = FALSE]), dim = 2)
      xz <- model$model_xz[[t]]$forward(sxz_in)
      xt <- xz[, 1:model$x_dim, drop = FALSE]
      zt <- xz[, (model$x_dim + 1):(model$x_dim + model$z_dim), drop = FALSE]
      if (model$x_binary) xt <- (xt > 0)$to(dtype = torch_float())
      if (model$z_binary) zt <- (zt > 0)$to(dtype = torch_float())
      x_all <- torch_cat(list(x_all, xt), dim = 2)
      z_all <- torch_cat(list(z_all, zt), dim = 2)

      sxz_p <- torch_cat(list(s, x_all, z_all), dim = 2)
      etat <- model$model_eta[[t]]$forward(sxz_p)
      sxeta <- torch_cat(list(s, x_all, etat), dim = 2)
      yt <- (model$model_y[[t]]$forward(sxeta) > 0.5)$to(dtype = torch_int32())
      y_all <- torch_cat(list(y_all, yt), dim = 2)
    }

    list(x = as.matrix(x_all), z = as.matrix(z_all), y = as.matrix(y_all))
  })
}

#' @param s Static baseline variables (matrix or tensor).
#' @param x Treatment matrix or tensor.
#' @param sample_size Number of samples (default: 100).
#' @rdname sample_causal_margin
#' @export
sample_causal_margin.frengression_surv <- function(model, s, x,
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
        yt <- (yt > 0.5)$to(dtype = torch_float())
        samples_t[[ss]] <- as.matrix(yt)
      }
      all_y[[t]] <- simplify2array(samples_t)
    }
    all_y
  })
}

#' @param s Static baseline variables (matrix or tensor).
#' @param x Treatment matrix or tensor.
#' @param sample_size Number of Monte Carlo samples (default: 100).
#' @param ... Additional arguments (ignored).
#' @rdname predict_causal
#' @export
predict_causal.frengression_surv <- function(model, s, x, sample_size = 100, ...) {
  y_cm <- sample_causal_margin.frengression_surv(model, s, x, sample_size)
  # Return event indicator
  lapply(y_cm, function(yt) (yt > 0) * 1)
}

#' @param s Static baseline variables (matrix or tensor).
#' @rdname predict.frengression
#' @export
predict.frengression_surv <- function(object, s, x, type = "mean",
                                       nsample = 200, ...) {
  predict_causal.frengression_surv(object, s, x, sample_size = nsample)
}
