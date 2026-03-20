#' Print a Frengression Model
#'
#' @param x A frengression object.
#' @param ... Additional arguments (ignored).
#' @export
print.frengression <- function(x, ...) {
  cat("\n frengression model")
  cat("\n   x_dim:", x$x_dim, " y_dim:", x$y_dim, " z_dim:", x$z_dim)
  cat("\n   noise_dim:", x$noise_dim, " hidden_dim:", x$hidden_dim, " num_layer:", x$num_layer)
  cat("\n   x_binary:", x$x_binary, " y_binary:", x$y_binary, " z_binary_dims:", x$z_binary_dims)
  if (!is.null(x$causal_fn)) cat("\n   causal margin: user-specified")
  if (!is.null(x$lossvec_y)) {
    cat("\n   trained: Y model (", nrow(x$lossvec_y), "iters)")
  }
  if (!is.null(x$lossvec_xz)) {
    cat("\n   trained: XZ model (", nrow(x$lossvec_xz), "iters)")
  }
  cat("\n")
}

#' Print a FrengressionSeq Model
#'
#' @param x A frengression_seq object.
#' @param ... Additional arguments (ignored).
#' @export
print.frengression_seq <- function(x, ...) {
  cat("\n frengression_seq model (longitudinal)")
  cat("\n   x_dim:", x$x_dim, " y_dim:", x$y_dim, " z_dim:", x$z_dim,
      " s_dim:", x$s_dim, " T:", x$T_steps)
  cat("\n   noise_dim:", x$noise_dim, " hidden_dim:", x$hidden_dim, " num_layer:", x$num_layer)
  cat("\n   x_binary:", x$x_binary, " y_binary:", x$y_binary, " z_binary:", x$z_binary)
  cat("\n")
}

#' Print a FrengressionSurv Model
#'
#' @param x A frengression_surv object.
#' @param ... Additional arguments (ignored).
#' @export
print.frengression_surv <- function(x, ...) {
  cat("\n frengression_surv model (survival)")
  cat("\n   x_dim:", x$x_dim, " y_dim:", x$y_dim, " z_dim:", x$z_dim,
      " s_dim:", x$s_dim, " T:", x$T_steps)
  cat("\n   noise_dim:", x$noise_dim, " hidden_dim:", x$hidden_dim, " num_layer:", x$num_layer)
  cat("\n   x_binary:", x$x_binary, " y_binary:", x$y_binary, " z_binary:", x$z_binary)
  cat("\n")
}
