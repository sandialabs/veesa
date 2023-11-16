#' @importFrom stats predict

compute_nmse <- function(x, y, f, n) {
  
  # Function for computing negative MSE ----------------------------------------
  
  # Inputs:
  #   x = dataset with n observations and p variables (training or testing)
  #   y = response variable associated with x (an n by k matrix is also allowed 
  #       for a multivariate (k responses) models trained using randomForestSRC)
  #   f = model to explain
  #   n = number of observations in x and y
  
  # Note: Multivariate version is computed by treating response variables as one vector 
  
  yhat = predict(f, x)
  if (is.null(dim(y))) {
    -sum((yhat - y) ^ 2) / n
  } else {
    k = length(yhat$regrOutput)
    yhat = map(.x = yhat$regrOutput, .f = function(x) x$predicted) %>% unlist()
    y = as.vector(y)
    - sum((yhat- y) ^ 2) / length(yhat)
  }

}