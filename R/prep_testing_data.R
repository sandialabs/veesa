#' Align test data and apply fPCA using elastic method applied to training data
#'
#' Applies steps 2 and 3 of the VEESA pipeline (alignment and elastic fPCA (jfpca, hfpca, or vfpca)) to 
#'     the testing data based on the training data prepared using "prep_training_data".
#'     
#' @param f Test data matrix (M x N) of N functions with M samples (already smoothed)
#' @param time Vector of size M describing the sample points
#' @param train_prep Object returned from applying "prep_training_data" to training data
#' @param omethod Method used for optimization when computing the Karcher mean (DP, DP2, RBFGS, or DPo)
#' 
#' @export prep_testing_data
#'
#' @importFrom fdasrvf f_to_srvf gradient optimum.reparam time_warping warp_f_gamma
#' @importFrom purrr map map2 pmap
#' 
#' @return List containing (varies slightly based on fpca method used): 
#' \itemize{
#'   \item time: vector of times when functions are observed (length of M)
#'   \item f0: original test data functions - matrix (M x N) of N functions with M samples
#'   \item fn: aligned test data functions - similar structure to f0
#'   \item q0: original test data SRSFs - similar structure to f0
#'   \item qn: aligned test data SRSFs - similar structure to f0
#'   \item mqn: training data SRSF mean (test data functions are aligned to this function)
#'   \item gam: test data warping functions - similar structure to f0
#'   \item coef: test data principal component coefficients
#'   \item psi: test data warping function SRVFs - similar structure to f0 (jfpca and hfpca only)
#'   \item nu: test data shooting functions - similar structure to f0 (jfpca and hfpca only)
#'   \item g: test data combination of aligned and shooting functions (jfpca only)
#' }

prep_testing_data <- function(f, time, train_prep, omethod = "DP") {

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
      method = omethod
    )
  
  # 4. Apply warping functions to align test data functions:
  fn = purrr::map2(.x = f, .y = gamma, .f = warp_f_gamma, time = time)

  # 5. Compute the SRSFs of the aligned functions
  qn = purrr::map(.x = fn, .f = f_to_srvf, time = time)

  #### Functional Principal Components ---------------------------------

  # 1. If applying jfpca or hfpca:
  if (fpca_type %in% c("jfpca", "hfpca")) {
    # Compute SRSFs of test data warping functions:
    psi <-
      purrr::map(.x = gamma, .f = fdasrvf::gradient, binsize = mean(diff(time))) %>%
      purrr::map(.f = sqrt)
    # Compute test data shooting functions:
    if (fpca_type == "jfpca") {
      mu_psi = fpca_train$mu_psi
    } else {
      mu_psi = rowMeans(matrix(unlist(psi), ncol = ntest, byrow = F)) 
    }
    nu = purrr::map(.x = psi, .f = fdasrvf:::inv_exp_map, Psi = mu_psi)
  }
  
  # 2. If applying jfpca or vfpca, obtain id value:
  if (fpca_type %in% c("jfpca", "vfpca")) {
    f_id = purrr::map(.x = fn, .f = fpca_train$id)
    q_id = purrr::map(.x = f_id, .f = function(f_id) sign(f_id) * sqrt(abs(f_id)))
  }
  
  # 3. Compute the principal components for the test data:
  if (fpca_type == "jfpca") {
    # First, create the vector g with aligned functions and shooting vectors:
    nu_scaled = purrr::map(.x = nu, .f = function(x) x * fpca_train$C)
    g = purrr::pmap(.l = list(qn, q_id, nu_scaled), .f = c)
    # Second, compute the PCs
    pcs = purrr::map(.x = g, .f = function(g) (g - fpca_train$mu_g) %*% fpca_train$U)
  } else if (fpca_type == "vfpca") {
    # First, join aligned functions with id value and their means
    h = purrr::pmap(.l = list(qn, q_id), .f = c)
    h_mean = c(q_mean_train, mean(unlist(q_id)))
    # Second, compute the PCs
    pcs = purrr::map(.x = h, .f = function(h) (h - h_mean) %*% fpca_train$U)
  } else if (fpca_type == "hfpca") {
    nu_mean = rowMeans(matrix(unlist(nu), ncol = ntest, byrow = F))
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
          unlist(x) %>% matrix(ncol = ntest, byrow = F)
        } else {
          x
        }
      }
    )

  # Return a list with the results
  return(res)

}
