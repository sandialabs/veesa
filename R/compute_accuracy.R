#' @importFrom stats predict

compute_accuracy <- function(x, y, f, n) {
  
  # Function for computing accuracy ----------------------------------------
  
  # Inputs: 
  #   x = dataset with n observations and p variables (training or testing)
  #   y = response variable associated with x
  #   f = model to explain
  #   n = number of observations in x and y
  
  yhat = predict(f, x)
  sum(yhat == y) / n

}