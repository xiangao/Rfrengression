#' Convert Data Frame to Numeric Matrix
#'
#' Converts a data frame into a numeric matrix. Factor and character
#' variables are converted to numeric.
#'
#' @param X A data frame to be converted.
#' @return A numeric matrix.
#' @keywords internal
dftomat <- function(X) {
  X <- data.frame(lapply(X, function(x) {
    if (is.factor(x)) {
      as.numeric(as.character(x))
    } else if (is.character(x)) {
      as.numeric(as.factor(x))
    } else {
      as.numeric(x)
    }
  }))
  as.matrix(X)
}

#' Ensure Input is a Torch Tensor
#'
#' @param x A matrix, data frame, vector, or torch tensor.
#' @param dtype Torch data type (default: torch_float()).
#' @return A torch tensor.
#' @keywords internal
as_tensor <- function(x, dtype = torch_float()) {
  if (inherits(x, "torch_tensor")) return(x)
  if (is.data.frame(x)) x <- dftomat(x)
  if (is.vector(x) && is.numeric(x)) x <- matrix(x, ncol = 1)
  torch_tensor(x, dtype = dtype)
}
