#' Align test data and apply fPCA using elastic method applied to training data
#'
#' Applies steps 2 and 3 of the VEESA pipeline (alignment and elastic fPCA
#'     (jfpca, hfpca, or vfpca)) to the testing data based on the training
#'     data prepared using "prep_training_data".
#'
#' @details The testing data are aligned to the training data centroid by
#'     penalized optimal reparameterization
#'     (`fdasrvf::optimum.reparam`). By default, the elasticity (`lambda`) and
#'     the penalty (`penalty_method`) used for the training data alignment are
#'     reused here, so that the testing data are aligned under the same
#'     criterion as the training data. They can be overridden with the
#'     `lambda` and `penalty_method` arguments. The penalty is weighted by
#'     `lambda`, so no penalty is applied when `lambda = 0` (the default in
#'     `prep_training_data`).
#'
#' @param f Matrix (size M x N) of test data with N functions and M samples.
#' @param time Vector of size M describing the sample points
#' @param train_prep Object returned from applying "prep_training_data" to
#'        training data.
#' @param optim_method Method used for optimization when computing the Karcher
#'        mean. "DP", "DPo", and "RBFGS".
#' @param lambda Numeric value specifying the elasticity used when aligning the
#'        testing data to the training data centroid. Default is `NULL`, which
#'        reuses the value of `lambda` that was applied to the training data by
#'        `prep_training_data` (stored in
#'        `train_prep$alignment$call$lambda`). Supplying a value overrides the
#'        training data value, but note that this makes the testing data
#'        alignment inconsistent with the training data alignment.
#' @param penalty_method A string specifying the penalty term used in the
#'   formulation of the cost function to minimize for alignment. Choices are
#'   `"roughness"` which uses the norm of the second derivative, `"l2gam"`
#'   which uses the \eqn{L^2} distance of the warping function to the identity,
#'   `"l2psi"` which uses the \eqn{L^2} distance of the SRVF of the warping
#'   function to that of the identity, `"geodesic"` which uses the geodesic
#'   distance to the identity, and `"none"` which applies no penalty. `"norm"`
#'   is kept for backward compatibility as an alias for `"l2gam"`. The penalty
#'   is weighted by `lambda`, so it has no effect when `lambda = 0`. Default is
#'   `NULL`, which reuses the penalty that was applied to the training data by
#'   `prep_training_data` (stored in
#'   `train_prep$alignment$call$penalty_method`).
#'
#' @export prep_testing_data
#'
#' @importFrom fdasrvf f_to_srvf gradient optimum.reparam time_warping warp_f_gamma
#' @importFrom purrr map map2 pmap
#'
#' @returns List containing (varies slightly based on fpca method used):
#' \itemize{
#'   \item time: vector of times when functions are observed (length of M)
#'   \item f0: original test data functions - matrix (M x N) of N functions
#'         with M samples
#'   \item fn: aligned test data functions - similar structure to f0
#'   \item q0: original test data SRSFs - similar structure to f0
#'   \item qn: aligned test data SRSFs - similar structure to f0
#'   \item mqn: training data SRSF mean (test data functions are aligned to
#'         this function)
#'   \item gam: test data warping functions - similar structure to f0
#'   \item coef: test data principal component coefficients - matrix with one
#'         row per principal component and one column per test function (note
#'         that this is the transpose of the layout used by the training data
#'         coefficients in `prep_training_data`)
#'   \item psi: test data warping function SRVFs - similar structure to f0
#'         (jfpca and hfpca only)
#'   \item nu: test data shooting functions - similar structure to f0 (jfpca
#'         and hfpca only)
#'   \item g: test data combination of aligned and shooting functions (jfpca
#'         only)
#'   \item call: list recording the alignment settings used (lambda,
#'         penalty_method, and optim_method)
#' }
#'
#' @examples
#' # Load packages
#' library(dplyr)
#' library(tidyr)
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
#' # Convert testing data to matrix
#' shifted_peaks_test_matrix <-
#'   shifted_peaks_sub |>
#'   filter(data == "Testing") |>
#'   select(-t) |>
#'   mutate(index = paste0("t", index)) |>
#'   pivot_wider(names_from = index, values_from = y) |>
#'   select(-data, -id, -group) |>
#'   as.matrix() |>
#'   t()
#'
#' # Obtain veesa pipeline testing data
#' veesa_test <- prep_testing_data(
#'   f = shifted_peaks_test_matrix,
#'   time = shifted_peaks_times,
#'   train_prep = veesa_train,
#'   optim_method = "DP"
#'  )

prep_testing_data <- function(
    f,
    time,
    train_prep,
    optim_method = "DP",
    lambda = NULL,
    penalty_method = NULL
  ) {

  #### Setup -----------------------------------------------------------

  # Determine the number of functions in the test data
  ntest = dim(f)[2]

  # Change times to be between 0 and 1
  time = seq(0, 1, length.out = length(time))

  # Convert f to a list
  f = purrr::map(.x = 1:ntest, .f = function(x) f[, x])

  # Separate train_prep into alignment and jfPCA parts
  aligned_train = train_prep$alignment
  fpca_train = train_prep$fpca_res
  fpca_type = train_prep$fpca_type

  # Determine the alignment penalty to apply to the test data. By default,
  # the elasticity and penalty used to align the training data (stored in the
  # 'call' component of the fdasrvf::time_warping output) are reused so that
  # the test data are aligned under the same criterion as the training data.
  train_call = aligned_train$call
  if (is.null(lambda)) {
    lambda = if (is.null(train_call$lambda)) 0 else train_call$lambda
  }
  if (!is.numeric(lambda) || length(lambda) != 1 || is.na(lambda)) {
    stop("lambda must be a single numeric value.")
  }
  if (is.null(penalty_method)) {
    penalty_method <-
      if (is.null(train_call$penalty_method)) {
        "roughness"
      } else {
        train_call$penalty_method
      }
  }
  penalty_method <-
    match.arg(
      arg = penalty_method,
      choices = c("roughness", "l2gam", "l2psi", "geodesic", "none", "norm")
    )
  # "norm" is an alias kept in fdasrvf for backward compatibility, but
  # fdasrvf::optimum.reparam only accepts the current penalty names
  if (penalty_method == "norm") penalty_method = "l2gam"

  # Note: This function performs all computation as lists and converts
  # the lists to matrices at the end before returning the results

  #### Functional Alignment --------------------------------------------

  # 1. Identify Karcher mean training data SRSFs:
  #q_mean_train = rowMeans(aligned_train$qn)
  q_mean_train = aligned_train$mqn

  # 2. Convert the test data to SRSFs (q_i's):
  q = purrr::map(.x = f, .f = fdasrvf::f_to_srvf, time = time)

  # 3. Obtain warping functions needed to align test data to training
  # data Karcher Mean:
  gamma <-
    purrr::map(
      .x = q,
      .f = fdasrvf::optimum.reparam,
      Q1 = q_mean_train,
      T1 = time,
      T2 = time,
      lambda = lambda,
      pen = penalty_method,
      method = optim_method
    )

  # 4. Apply warping functions to align test data functions:
  fn = purrr::map2(.x = f, .y = gamma, .f = fdasrvf::warp_f_gamma, time = time)

  # 5. Compute the SRSFs of the aligned functions
  qn = purrr::map(.x = fn, .f = fdasrvf::f_to_srvf, time = time)

  #### Functional Principal Components ---------------------------------

  # 1. If applying jfpca or hfpca:
  if (fpca_type %in% c("jfpca", "hfpca")) {
    # Compute SRSFs of test data warping functions:
    psi <-
      purrr::map(
        .x = gamma,
        .f = fdasrvf::gradient,
        binsize = mean(diff(time))
      ) %>%
      purrr::map(.f = sqrt)
    # Compute test data shooting functions (the base point must be the one
    # estimated from the training data so that the test shooting vectors live
    # in the same tangent space as the training ones):
    if (fpca_type == "jfpca") {
      mu_psi = fpca_train$mu_psi
    } else {
      mu_psi = fpca_train$mu
    }
    nu = purrr::map(.x = psi, .f = fdasrvf::inv_exp_map, Psi = mu_psi)
  }

  # 2. If applying jfpca or vfpca, obtain id value:
  if (fpca_type %in% c("jfpca", "vfpca")) {
    f_id = purrr::map(.x = fn, .f = fpca_train$id)
    q_id = purrr::map(
      .x = f_id,
      .f = function(f_id) sign(f_id) * sqrt(abs(f_id))
    )
  }

  # 3. Compute the principal components for the test data:
  if (fpca_type == "jfpca") {
    # First, create the vector g with aligned functions and shooting vectors:
    nu_scaled = purrr::map(.x = nu, .f = function(x) x * fpca_train$C)
    g = purrr::pmap(.l = list(qn, q_id, nu_scaled), .f = c)
    # Second, compute the PCs
    pcs = purrr::map(.x = g, .f = function(g) (g - fpca_train$mu_g) %*% fpca_train$U)
  } else if (fpca_type == "vfpca") {
    # First, join aligned functions with id value
    h = purrr::pmap(.l = list(qn, q_id), .f = c)
    # Second, compute the PCs (centered using the training data mean stored by
    # fdasrvf::vertFPCA, not a mean recomputed from the test data)
    h_mean = fpca_train$mqn
    pcs = purrr::map(.x = h, .f = function(h) (h - h_mean) %*% fpca_train$U)
  } else if (fpca_type == "hfpca") {
    # Centered using the training data shooting vector mean stored by
    # fdasrvf::horizFPCA, not a mean recomputed from the test data
    nu_mean = fpca_train$vm
    pcs = purrr::map(.x = nu, .f = function(nu) (nu - nu_mean) %*% fpca_train$U)
  }

  #### Output ----------------------------------------------------------

  # Put all results in a list
  res_list <-
    list(
      time = time,
      f0 = f,
      fn = fn,
      q0 = q,
      qn = qn,
      mqn = q_mean_train,
      gam = gamma,
      coef = pcs
    )
  # Add additional items for hfpca and jfpca
  if (fpca_type == "hfpca") {
    res_list = res_list %>% append(list(psi = psi, nu = nu))
  } else if (fpca_type == "jfpca") {
    res_list = res_list %>% append(list(psi = psi, nu = nu, g = g))
  }

  # Convert all lists to matrices
  res <-
    res_list %>%
    map(
      .f = function(x) {
        if (typeof(x) == "list") {
          unlist(x) %>% matrix(ncol = ntest, byrow = FALSE)
        } else {
          x
        }
      }
    )

  # Record the alignment settings that were used (mirrors the 'call'
  # component of the fdasrvf::time_warping output)
  res$call <-
    list(
      lambda = lambda,
      penalty_method = penalty_method,
      optim_method = optim_method
    )

  # Return a list with the results
  return(res)

}
