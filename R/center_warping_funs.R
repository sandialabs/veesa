#' Center warping functions
#'
#' The function 'prep_training_data' does not center the warping functions. For
#' visualizing the aligned and warping functions, it can be easier to look at
#' centered versions. This function centers the warping functions and
#' corresponding aligned functions.
#'
#' @param train_obj Output object from `prep_training_data`
#'
#' @importFrom stats approx
#'
#' @return Object with the same structure as `train_obj` but qn, fn, and gam
#'         have been replaced by centered versions

center_warping_funs <- function(train_obj) {

  # Extract and prepare objects based on training data
  aligned = train_obj$alignment
  time = aligned$time
  gam = t(aligned$warping_functions)
  N = nrow(gam)
  M = ncol(gam)
  mq = aligned$mqn

  # Compute gamma inverse
  gamI = fdasrvf::SqrtMeanInverse(t(gam))
  gamI_dev = fdasrvf::gradient(gamI, 1/(M-1))

  # Apply centering
  time0 = (time[length(time)] - time[1]) * gamI + time[1]
  mq = approx(time, mq, xout = time0)$y * sqrt(gamI_dev)
  for (k in 1:N){
    aligned$qn[,k] = approx(time, aligned$qn[,k], xout = time0)$y * sqrt(gamI_dev)
    aligned$fn[,k] = approx(time, aligned$fn[,k], xout = time0)$y
    aligned$warping_functions[,k] = approx(time, gam[k,], xout = time0)$y
  }

  # Return the centered results
  train_obj$alignment = aligned
  return(train_obj)

}
