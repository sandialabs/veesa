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
  
  probs = predict(f, x, type = "prob")
  yref = attr(probs, which = "dimnames")[[2]][1] %>% as.numeric()
  yind = ifelse(y == yref, 1, 0)
  prob1 = purrr::map_dbl(probs[,1], .f = function(prob) max(eps, min(1 - eps, prob)))
  mean((yind * log(prob1)) + ((1 - yind) * log(1 - prob1)))
  
}
