#' Compute permutation feature importance (PFI)
#'
#' Function for computing PFI for a given model and dataset (training or
#' testing)
#'
#' @param x Dataset with n observations and p variables (training or testing)
#' @param y Response variable (or matrix) associated with x
#' @param f Model to explain
#' @param K Number of repetitions to perform for PFI
#' @param metric Metric used to compute PFI (choose from "accuracy", "logloss",
#'        and "nmse")
#' @param eps Log loss is undefined for p = 0 or p = 1, so probabilities are
#         clipped to max(eps, min(1 - eps, p)). Default is 1e-15.
#'
#' @export compute_pfi
#'
#' @importFrom dplyr %>%
#' @importFrom purrr map2
#' @importFrom stats setNames
#'
#' @returns List containing
#' \itemize{
#'   \item \code{pfi}: Vector of PFI values (averaged over replicates)
#'   \item \code{pfi_single_reps}: Matrix of containing the feature importance
#'          values from each replicate (rows associated with reps; columns
#'          associated with data observations)
#' }
#'
#' @examples
#' # Load packages
#' library(dplyr)
#' library(tidyr)
#' library(randomForest)
#'
#' # Select a subset of functions from shifted peaks data
#' sub_ids <-
#'   shifted_peaks$data |>
#'   select(data, group, id) |>
#'   distinct() |>
#'   group_by(data, group) |>
#'   slice(1:4) |>
#'   ungroup()
#'
#' # Create a smaller version of shifted data
#' shifted_peaks_sub <-
#'   shifted_peaks$data |>
#'   filter(id %in% sub_ids$id)
#'
#' # Extract times
#' shifted_peaks_times = unique(shifted_peaks_sub$t)
#'
#' # Convert training data to matrix
#' shifted_peaks_train_matrix <-
#'   shifted_peaks_sub |>
#'   filter(data == "Training") |>
#'   select(-t) |>
#'   mutate(index = paste0("t", index)) |>
#'   pivot_wider(names_from = index, values_from = y) |>
#'   select(-data, -id, -group) |>
#'   as.matrix() |>
#'   t()
#'
#' # Obtain veesa pipeline training data
#' veesa_train <-
#'   prep_training_data(
#'     f = shifted_peaks_train_matrix,
#'     time = shifted_peaks_times,
#'     fpca_method = "jfpca"
#'   )
#'
#' # Obtain response variable values
#' model_output <-
#'   shifted_peaks_sub |>
#'   filter(data == "Training") |>
#'   select(id, group) |>
#'   distinct()
#'
#' # Prepare data for model
#' model_data <-
#'   veesa_train$fpca_res$coef |>
#'   data.frame() |>
#'   mutate(group = factor(model_output$group))
#'
#' # Train model
#' set.seed(20210301)
#' rf <-
#'   randomForest(
#'     formula = group ~ .,
#'     data = model_data
#'   )
#'
#' # Compute feature importance values
#' pfi <-
#'   compute_pfi(
#'     x = model_data |> select(-group),
#'     y = model_data$group,
#'     f = rf,
#'     K = 1,
#'     metric = "accuracy"
#'  )

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
      xtildep = sample(x[, p], size = n, replace = FALSE)
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
    matrix(nrow = K, ncol = p, byrow = TRUE)

  # Compute PFI
  list(pfi = m - (colSums(mpk_matrix) / K), pfi_single_reps = m - mpk_matrix)

}
