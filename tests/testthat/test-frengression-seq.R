skip_if_not(torch::torch_is_installed(), "torch backend not available")

.make_longitudinal_data <- function(n = 150, T_steps = 3, seed = 42) {
  set.seed(seed)
  s <- matrix(rbinom(n, 1, 0.5), ncol = 1)
  x_all <- matrix(0, n, T_steps)
  z_all <- matrix(0, n, T_steps)
  y_all <- matrix(0, n, T_steps)

  z_all[, 1] <- rnorm(n, mean = 0.3 * s)
  x_all[, 1] <- rbinom(n, 1, plogis(0.5 * z_all[, 1] + 0.2 * s))
  y_all[, 1] <- rbinom(n, 1, plogis(-2 + 0.5 * x_all[, 1] + 0.3 * z_all[, 1]))

  for (t in 2:T_steps) {
    z_all[, t] <- rnorm(n, mean = 0.3 * s + 0.2 * z_all[, t - 1])
    x_all[, t] <- rbinom(n, 1, plogis(0.5 * z_all[, t] + 0.2 * s))
    y_all[, t] <- rbinom(n, 1, plogis(-2 + 0.5 * x_all[, t] +
                                          0.3 * z_all[, t] + 0.1 * x_all[, t - 1]))
  }
  list(s = s, x = x_all, z = z_all, y = y_all, T_steps = T_steps)
}

test_that("frengression_seq constructor builds a valid model object", {
  d <- .make_longitudinal_data()
  model <- frengression_seq(x_dim = 1, y_dim = 1, z_dim = 1, T_steps = d$T_steps,
                             s_dim = 1, noise_dim = 3, num_layer = 2, hidden_dim = 8,
                             x_binary = TRUE, y_binary = TRUE)
  expect_s3_class(model, "frengression_seq")
  expect_equal(model$T_steps, d$T_steps)
  expect_length(model$model_xz, d$T_steps)
  expect_output(print(model), "frengression_seq")
})

test_that("the three-stage training pipeline (xz, e, y) runs end to end", {
  d <- .make_longitudinal_data()
  model <- frengression_seq(x_dim = 1, y_dim = 1, z_dim = 1, T_steps = d$T_steps,
                             s_dim = 1, noise_dim = 3, num_layer = 2, hidden_dim = 8,
                             x_binary = TRUE, y_binary = TRUE, s_in_predict = TRUE)

  model <- train_xz(model, s = d$s, x = d$x, z = d$z, num_iters = 5, lr = 1e-3, silent = TRUE)
  model <- train_e(model, s = d$s, x = d$x, z = d$z, num_iters = 5, lr = 1e-3, silent = TRUE)
  model <- train_y(model, s = d$s, x = d$x, z = d$z, y = d$y, num_iters = 5, lr = 1e-3, silent = TRUE)

  expect_s3_class(model, "frengression_seq")
})

test_that("sample_joint returns correctly shaped, finite (X, Z, Y) trajectories", {
  d <- .make_longitudinal_data()
  model <- frengression_seq(x_dim = 1, y_dim = 1, z_dim = 1, T_steps = d$T_steps,
                             s_dim = 1, noise_dim = 3, num_layer = 2, hidden_dim = 8,
                             x_binary = TRUE, y_binary = TRUE)
  model <- train_xz(model, s = d$s, x = d$x, z = d$z, num_iters = 5, lr = 1e-3, silent = TRUE)
  model <- train_e(model, s = d$s, x = d$x, z = d$z, num_iters = 5, lr = 1e-3, silent = TRUE)
  model <- train_y(model, s = d$s, x = d$x, z = d$z, y = d$y, num_iters = 5, lr = 1e-3, silent = TRUE)

  joint <- sample_joint(model, s = d$s[1:40, , drop = FALSE])
  expect_equal(dim(joint$x), c(40, d$T_steps))
  expect_equal(dim(joint$z), c(40, d$T_steps))
  expect_equal(dim(joint$y), c(40, d$T_steps))
  expect_true(all(is.finite(joint$x)) && all(is.finite(joint$z)) && all(is.finite(joint$y)))
})

test_that("predict_causal returns one finite matrix per time step under a fixed regime", {
  d <- .make_longitudinal_data()
  model <- frengression_seq(x_dim = 1, y_dim = 1, z_dim = 1, T_steps = d$T_steps,
                             s_dim = 1, noise_dim = 3, num_layer = 2, hidden_dim = 8,
                             x_binary = TRUE, y_binary = TRUE, s_in_predict = TRUE)
  model <- train_xz(model, s = d$s, x = d$x, z = d$z, num_iters = 5, lr = 1e-3, silent = TRUE)
  model <- train_e(model, s = d$s, x = d$x, z = d$z, num_iters = 5, lr = 1e-3, silent = TRUE)
  model <- train_y(model, s = d$s, x = d$x, z = d$z, y = d$y, num_iters = 5, lr = 1e-3, silent = TRUE)

  s_test <- matrix(c(0, 1), ncol = 1)
  x_always <- matrix(1, nrow = 2, ncol = d$T_steps)

  y_always <- predict_causal(model, s = s_test, x = x_always, target = "mean", sample_size = 20)
  expect_type(y_always, "list")
  expect_length(y_always, d$T_steps)
  for (yt in y_always) {
    expect_equal(nrow(yt), 2)
    expect_true(all(is.finite(yt)))
  }
})
