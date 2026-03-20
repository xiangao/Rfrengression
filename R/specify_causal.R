#' Specify a Custom Causal Margin
#'
#' Replaces the learned causal margin P(Y|do(X)) with a user-specified
#' function. This allows generating joint data with a known causal effect
#' while preserving the learned marginal P(X,Z) and association structure.
#'
#' @param model A frengression or frengression_seq object.
#' @param causal_fn A function taking (x, eta) tensors and returning a
#'   y tensor. Here x is the treatment (batch x x_dim) and eta is the
#'   noise input (batch x y_dim) drawn from N(0,I).
#' @return The model with the custom causal margin set.
#'
#' @examples
#' \donttest{
#'   model <- frengression(x_dim = 1, y_dim = 1, z_dim = 1)
#'   # Linear causal effect: Y = 5*X + noise
#'   model <- specify_causal(model, function(x, eta) 5 * x + eta)
#' }
#'
#' @export
specify_causal <- function(model, causal_fn) {
  model$causal_fn <- causal_fn
  model
}

#' Reset Outcome Models
#'
#' Reinitializes model_y and model_eta with fresh random weights.
#' Also clears any user-specified causal margin.
#'
#' @param model A frengression object.
#' @param ... Additional arguments (ignored).
#' @return The model with reinitialized outcome networks.
#' @export
reset_y_models <- function(model, ...) UseMethod("reset_y_models")

#' @rdname reset_y_models
#' @export
reset_y_models.frengression <- function(model, ...) {
  out_act <- if (model$y_binary) "sigmoid" else NULL
  model$model_y <- build_sto_net(model$x_dim + model$y_dim, model$y_dim,
                                 model$num_layer, model$hidden_dim,
                                 model$noise_dim, out_act = out_act)
  model$model_eta <- build_sto_net(model$x_dim + model$z_dim, model$y_dim,
                                   model$num_layer, model$hidden_dim,
                                   model$noise_dim)
  model$causal_fn <- NULL
  model$lossvec_y <- NULL
  model
}
