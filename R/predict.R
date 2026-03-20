#' Predict Causal Effects
#'
#' Computes predictions from the causal margin P(Y|do(X=x)), returning
#' the mean, quantiles, or raw samples. For the learned model, this
#' marginalizes over the noise input eta ~ N(0,I) that substitutes for
#' the confounders.
#'
#' @param object A frengression object.
#' @param x Intervention values for X (matrix, vector, or data frame).
#' @param type Type of prediction: "mean", "quantile", or "sample" (default: "mean").
#' @param nsample Number of Monte Carlo samples (default: 200).
#' @param quantiles Quantile levels if type = "quantile" (default: seq(0.1, 0.9, 0.1)).
#' @param trim Trimming proportion for mean (default: 0.05).
#' @param ... Additional arguments (ignored).
#' @return A matrix of predictions.
#'
#' @export
predict.frengression <- function(object, x, type = c("mean", "sample", "quantile"),
                                  nsample = 200, quantiles = seq(0.1, 0.9, 0.1),
                                  trim = 0.05, ...) {
  type <- match.arg(type)
  x <- as_tensor(x)
  batch_size <- x$size(1)

  with_no_grad({
    samp_list <- list()
    for (s in seq_len(nsample)) {
      if (is.null(object$causal_fn)) {
        # Generate random eta and pass (x, eta) to model_y
        eta <- torch_randn(c(batch_size, object$y_dim))
        x_eta <- torch_cat(list(x, eta), dim = 2)
        samp_list[[s]] <- object$model_y$forward(x_eta)
      } else {
        eta <- torch_randn(c(batch_size, object$y_dim))
        samp_list[[s]] <- object$causal_fn(x, eta)
      }
    }
    samp <- torch_stack(samp_list, dim = 3)  # batch x y_dim x nsample

    if (object$y_binary) {
      samp <- (samp > 0.5)$to(dtype = torch_float())
    }

    samp_arr <- as.array(samp)
  })

  if (type == "sample") return(samp_arr)

  if (type == "mean") {
    return(apply(samp_arr, c(1, 2), mean, trim = trim))
  }

  if (type == "quantile") {
    out <- apply(samp_arr, c(1, 2), quantile, probs = quantiles)
    if (length(quantiles) == 1) {
      return(matrix(out, nrow = dim(samp_arr)[1]))
    }
    return(aperm(out, c(2, 3, 1)))
  }
}

#' Predict Causal Effects (generic)
#'
#' @param model A frengression object.
#' @param ... Additional arguments passed to methods.
#' @return A matrix of predictions.
#' @export
predict_causal <- function(model, ...) UseMethod("predict_causal")

#' @param x Intervention values.
#' @param target "mean" or "sample" (default: "mean").
#' @param sample_size Number of Monte Carlo samples (default: 100).
#' @rdname predict_causal
#' @export
predict_causal.frengression <- function(model, x, target = "mean",
                                         sample_size = 100, ...) {
  predict.frengression(model, x, type = target, nsample = sample_size)
}
