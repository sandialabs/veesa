#' Align training data and apply a method of elastic fPCA
#'
#' Applies steps 2 and 3 of the VEESA pipeline (alignment and elastic fPCA) to the training data
#'     in preparation for inputting the data to the model in step 4.
#'
#' @param f Training data matrix (M x N) of N functions with M samples (already smoothed)
#' @param time Vector of size M describing the sample points
#' @param fpca_method Character string specifying the type of elastic fPCA method to use ('jfpca', 'hfpca', or 'vfpca')
#' @param ci Geodesic standard deviations to be computed (default = c(-2, -1, 0, 1, 2))
#' @param optim_method Method used for optimization when computing the Karcher mean (DP,DP2,RBFGS,DPo)
#'
#' @export prep_training_data
#'
#' @importFrom fdasrvf f_to_srvf horizFPCA jointFPCA vertFPCA optimum.reparam time_warping warp_f_gamma
#' @importFrom purrr map map2
#'
#' @return List with three objects:
#' \itemize{
#'   \item alignment: output from fdasrvf::time_warping
#'   \item fpca_type: type of elastic FPCA method applied
#'   \item fpca_res: output from fdasrvf::jointFPCA, fdasrvf::horizFPCA, or fdasrvf::vertFPCA (dependent on fpca_type)
#' }

prep_training_data <- function(f, time, fpca_method, ci = c(-2, -1, 0, 1, 2), optim_method = "DP") {

  # Make sure 'fpca_method' is all lower case
  fpca_method = tolower(fpca_method)

  # Compute number of training functions
  #N = dim(f)[2]

  # Change times to be between 0 and 1
  time = seq(0, 1, length.out = length(time))

  # Apply time_warping from fdasrvf to training data
  # Note: We set the centering to false so that later on, the test data
  # transformations also also implemented with no centering to achieve
  # better numerical estimates
  aligned <-
    fdasrvf::time_warping(
      f = f,
      time = time,
      parallel = T,
      optim_method = optim_method,
      center = F
    )

  # Apply an elastic fPCA to the adjusted aligned training data using fdasrvf
  if (fpca_method == "jfpca") {
    fpca_res = fdasrvf::jointFPCA(
      warp_data = aligned,
      no = length(time),
      ci = ci,
      showplot = F
    )
  } else if (fpca_method == "hfpca") {
    fpca_res = fdasrvf::horizFPCA(
      warp_data = aligned,
      no = length(time),
      ci = ci,
      showplot = F
    )
  } else if (fpca_method == "vfpca") {
    fpca_res = fdasrvf::vertFPCA(
      warp_data = aligned,
      no = length(time),
      ci = ci,
      showplot = F
    )
  } else {
    stop("fpca_method specified incorrectly. Must be 'jfpca', 'vfpca', or 'hfpca'.")
  }

  # Return the results from alignment and elastic fPCA
  list(alignment = aligned, fpca_type = fpca_method, fpca_res = fpca_res)

}
