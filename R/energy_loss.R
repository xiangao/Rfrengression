#' Two-Sample Energy Loss
#'
#' Computes the energy loss between observed values and two independent
#' stochastic predictions. The loss is E|Y-Yhat| - 0.5*E|Yhat-Yhat'|.
#'
#' @param yt Tensor of observed values.
#' @param mxt Tensor of first stochastic predictions.
#' @param mxpt Tensor of second stochastic predictions.
#' @return A list with components: loss (total), loss1 (prediction error),
#'   loss2 (diversity term).
#' @keywords internal
energy_loss <- function(yt, mxt, mxpt) {
  s1 <- torch_mean(torch_norm(yt - mxt, 2, dim = 2)) / 2 +
    torch_mean(torch_norm(yt - mxpt, 2, dim = 2)) / 2
  s2 <- torch_mean(torch_norm(mxt - mxpt, 2, dim = 2))
  list(loss = s1 - s2 / 2, loss1 = s1, loss2 = s2)
}
