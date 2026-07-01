skip_if_not(torch::torch_is_installed(), "torch backend not available")

test_that("frengression constructor builds a valid model object", {
  model <- frengression(x_dim = 1, y_dim = 1, z_dim = 2, num_layer = 2,
                         hidden_dim = 8, noise_dim = 3)
  expect_s3_class(model, "frengression")
  expect_equal(model$x_dim, 1)
  expect_equal(model$y_dim, 1)
  expect_equal(model$z_dim, 2)
  expect_output(print(model), "frengression")
})

test_that("train_y and train_xz run without error and produce finite losses", {
  set.seed(1)
  n <- 100
  z <- matrix(rnorm(n), ncol = 1)
  x <- matrix(rnorm(n), ncol = 1) + z
  y <- 2 * x + z + matrix(rnorm(n, sd = 0.1), ncol = 1)

  model <- frengression(x_dim = 1, y_dim = 1, z_dim = 1, num_layer = 2,
                         hidden_dim = 8, noise_dim = 3)

  model <- train_y(model, x, z, y, num_iters = 10, lr = 1e-3, silent = TRUE)
  model <- train_xz(model, x, z, num_iters = 10, lr = 1e-3, silent = TRUE)

  expect_true(all(is.finite(model$lossvec_y)))
  expect_true(all(is.finite(model$lossvec_xz)))
})

test_that("sample_joint, sample_xz, and sample_causal_margin return correctly shaped, finite output", {
  set.seed(2)
  n <- 100
  z <- matrix(rnorm(n), ncol = 1)
  x <- matrix(rnorm(n), ncol = 1)
  y <- matrix(rnorm(n), ncol = 1)

  model <- frengression(x_dim = 1, y_dim = 1, z_dim = 1, num_layer = 2,
                         hidden_dim = 8, noise_dim = 3)
  model <- train_y(model, x, z, y, num_iters = 5, lr = 1e-3, silent = TRUE)
  model <- train_xz(model, x, z, num_iters = 5, lr = 1e-3, silent = TRUE)

  joint <- sample_joint(model, sample_size = 50)
  expect_equal(dim(joint$x), c(50, 1))
  expect_equal(dim(joint$y), c(50, 1))
  expect_equal(dim(joint$z), c(50, 1))
  expect_true(all(is.finite(joint$x)) && all(is.finite(joint$y)) && all(is.finite(joint$z)))

  xz <- sample_xz(model, sample_size = 30)
  expect_equal(dim(xz$x), c(30, 1))
  expect_equal(dim(xz$z), c(30, 1))

  margin <- sample_causal_margin(model, matrix(1, nrow = 20, ncol = 1), sample_size = 20)
  # simplify2array stacks the `sample_size` (batch x y_dim) draws along a 3rd
  # dimension: (batch, y_dim, sample_size), NOT a (batch, sample_size) matrix.
  expect_equal(dim(margin), c(20, 1, 20))
  expect_true(all(is.finite(margin)))
})

test_that("predict() mean/sample/quantile types run and agree in scale", {
  set.seed(3)
  n <- 100
  z <- matrix(rnorm(n), ncol = 1)
  x <- matrix(rnorm(n), ncol = 1)
  y <- 2 * x + matrix(rnorm(n, sd = 0.1), ncol = 1)

  model <- frengression(x_dim = 1, y_dim = 1, z_dim = 1, num_layer = 2,
                         hidden_dim = 8, noise_dim = 3)
  model <- train_y(model, x, z, y, num_iters = 20, lr = 1e-2, silent = TRUE)

  x_new <- matrix(c(-1, 0, 1), ncol = 1)
  pred_mean <- predict(model, x_new, type = "mean", nsample = 200)
  expect_equal(length(pred_mean), 3)
  expect_true(all(is.finite(pred_mean)))

  pred_sample <- predict(model, x_new, type = "sample", nsample = 50)
  expect_true(all(is.finite(pred_sample)))
})

test_that("frengression recovers the sign and rough magnitude of a known binary ATE", {
  # Same DGP and training settings as vignette 'binary.Rmd': true ATE = 2.
  # This is a loose, non-flaky regression check (order-of-magnitude + correct
  # sign), not a convergence guarantee — it exists to catch the model class
  # silently breaking (e.g. a sign flip or dimension bug), not to validate
  # the paper's statistical claims.
  set.seed(123)
  n <- 3000
  z <- matrix(rnorm(n), ncol = 1)
  prob_x <- plogis(z)
  x <- matrix(rbinom(n, 1, prob_x), ncol = 1)
  y <- 2 * x + z + matrix(rnorm(n, sd = 0.5), ncol = 1)

  model <- frengression(x_dim = 1, y_dim = 1, z_dim = 1,
                         x_binary = TRUE, noise_dim = 5)
  model <- train_y(model, x, z, y, num_iters = 500, lr = 1e-3, silent = TRUE)
  model <- train_xz(model, x, z, num_iters = 500, lr = 1e-4, silent = TRUE)

  y1 <- predict(model, matrix(1, ncol = 1), type = "mean", nsample = 1000)
  y0 <- predict(model, matrix(0, ncol = 1), type = "mean", nsample = 1000)
  ate <- as.numeric(y1 - y0)

  expect_true(ate > 0.5 && ate < 3.5)
})
