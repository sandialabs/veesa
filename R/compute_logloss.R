#' @importFrom purrr map_dbl
#' @importFrom stats predict
 
compute_logloss <- function(x, y, f, eps) {
  
  # Function for computing log-loss ----------------------------------------
  
  # Inputs: 
  #   x = dataset with n observations and p variables (training or testing)
  #   y = response variable associated with x
  #   f = model to explain
  #   eps = Log loss is undefined for p = 0 or p = 1, so probabilities are 
  #         clipped to max(eps, min(1 - eps, p)). Default is 1e-15.
  
  # Note: This returns the mean log-likelihood, i.e. the negative of the log
  #       loss, so that (like accuracy and negative MSE) larger values are
  #       better and the differences computed by compute_pfi have the sign
  #       expected of a feature importance value.
  
  probs = predict(f, x, type = "prob")
  if (ncol(probs) != 2) {
    stop("'logloss' is only implemented for binary responses.")
  }
  yref = colnames(probs)[1]
  yind = ifelse(as.character(y) == yref, 1, 0)
  prob1 = purrr::map_dbl(probs[,1], .f = function(prob) max(eps, min(1 - eps, prob)))
  mean((yind * log(prob1)) + ((1 - yind) * log(1 - prob1)))
  
}
