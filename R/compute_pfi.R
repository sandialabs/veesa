#' Compute permutation feature importance (PFI)
#'
#' Function for computing PFI for a given model and dataset (training or testing)
#'     
#' @param x Dataset with n observations and p variables (training or testing)
#' @param y Response variable (or matrix) associated with x
#' @param f Model to explain
#' @param K Number of repetitions to perform for PFI
#' @param metric Metric used to compute PFI (choose from "accuracy", "logloss", and "nmse") 
#' @param eps Log loss is undefined for p = 0 or p = 1, so probabilities are 
#         clipped to max(eps, min(1 - eps, p)). Default is 1e-15.
#'  
#' @export compute_pfi
#'
#' @importFrom dplyr %>%
#' @importFrom purrr map2
#' @importFrom stats setNames
#' 
#' @return xxx

compute_pfi <- function(x, y, f, K, metric, eps = 1e-15) {
  
  # Make sure x is a data frame
  x = data.frame(x)
  
  # Determine number of observations and variables
  n = ifelse(is.null(dim(y)), length(y), dim(y)[1])
  p = dim(x)[2]
  
  # Check to make sure that the dimensions of x and y match
  if (n != dim(x)[1]) stop("Number of observations in x and y disagree.")
  
  # Compute metric on the observed data
  if (metric == "accuracy") {
    m = compute_accuracy(x = x, y = y, f = f, n = n)
  } else if (metric == "logloss") {
    m = compute_logloss(x = x, y = y, f = f, eps = eps)
  } else if (metric == "nmse") {
    m = compute_nmse(x = x, y = y, f = f, n = n)
  } else {
    stop("'metric' specified incorrectly. Select either 'accuracy', 'logloss', or 'nmse'.")
  }
  
  # Create a grid all combinations of variable number and repetition number
  pK = expand.grid(1:p, 1:K)
  pK = setNames(pK, c("p", "K"))
  
  # Compute metric when a variable is permuted for each var and rep
  mpk_matrix <- purrr::map2(
    .x = pK$p,
    .y = pK$K,
    .f = function(p, K) {
      xtildep = sample(x[, p], n, F)
      xtilde = x
      xtilde[, p] = xtildep
      if (metric == "accuracy") {
        mpk = compute_accuracy(x = xtilde, y = y, f = f, n = n)
      } else if (metric == "logloss") {
        mpk = compute_logloss(x = xtilde, y = y, f = f, eps = eps)
      } else if (metric == "nmse") {
        mpk = compute_nmse(x = xtilde, y = y, f = f, n = n)
      }
      mpk
    }
  ) %>%
    unlist() %>%
    matrix(nrow = K, ncol = p, byrow = T)
  
  # Compute PFI
  list(pfi = m - (colSums(mpk_matrix) / K), pfi_single_reps = m - mpk_matrix)
  
}
