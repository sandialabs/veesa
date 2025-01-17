#' Align training data and apply a method of elastic fPCA
#'
#' Applies steps 2 and 3 of the VEESA pipeline (alignment and elastic fPCA) to the training data
#'     in preparation for inputting the data to the model in step 4.
#'
#' @param f Matrix (size M x N) of training data with N functions and M samples.
#' @param time Vector of size M corresponding to the M sample points.
#' @param fpca_method Character string specifying the type of elastic fPCA
#'        method to use. Options are 'jfpca', 'hfpca', or 'vfpca'.
#' @param lambda Numeric value specifying the elasticity. Default is 0.
#' @param penalty_method String specifying the penalty term used in the
#'        formulation of the cost function to minimize for alignment. Choices
#'        are "roughness" which uses the norm of the second derivative,
#'        "geodesic" which uses the geodesic distance to the identity and
#'        "norm" which uses the Euclidean distance to the identity. Defaults
#'        is "roughness".
#' @param centroid_type String specifying the type of centroid to align to.
#'        Options are "mean" or "median". Defaults is "mean".
#' @param center_warpings Boolean specifying whether to center the estimated
#'        warping functions. Defaults is TRUE.
#' @param parallel Boolean specifying whether to run calculations in parallel.
#'        Defaults is FALSE.
#' @param cores Integer specifying the number of cores in parallel. Default is
#'        -1, which uses all cores.
#' @param optim_method Method used for optimization when computing the Karcher
#'        mean. Options are "DP", "DPo", and "RBFGS".
#' @param max_iter An integer value specifying the maximum number of iterations.
#'        Defaults to 20L.
#' @param id Integration point for f0. Default is midpoint.
#' @param C Balance value. Default = NULL.
#' @param ci Geodesic standard deviations to be computed. Default is
#'        c(-2, -1, 0, 1, 2).
#'
#' @export prep_training_data
#'
#' @importFrom fdasrvf f_to_srvf horizFPCA jointFPCA vertFPCA optimum.reparam
#'             time_warping warp_f_gamma
#' @importFrom purrr map map2
#'
#' @returns List with three objects:
#' \itemize{
#'   \item alignment: output from fdasrvf::time_warping
#'   \item fpca_type: type of elastic FPCA method applied
#'   \item fpca_res: output from fdasrvf::jointFPCA, fdasrvf::horizFPCA, or
#'         fdasrvf::vertFPCA (dependent on fpca_type)
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

prep_training_data <- function(
    f,
    time,
    fpca_method,
    lambda = 0,
    penalty_method = c("roughness", "geodesic", "norm"),
    centroid_type = c("mean", "median"),
    center_warpings = TRUE,
    parallel = FALSE,
    cores = -1,
    optim_method = c("DP", "DPo", "DP2", "RBFGS"),
    max_iter = 20L,
    id = NULL,
    C = NULL,
    ci = c(-2, -1, 0, 1, 2)
  ) {

  # Make sure 'fpca_method' is all lower case
  fpca_method = tolower(fpca_method)

  # Change times to be between 0 and 1
  time = seq(0, 1, length.out = length(time))

  # Apply time_warping from fdasrvf to training data
  aligned <-
    fdasrvf::time_warping(
      f = f,
      time = time,
      lambda = lambda,
      penalty_method = penalty_method,
      centroid_type = centroid_type,
      center_warpings = center_warpings,
      parallel = parallel,
      cores = cores,
      optim_method = optim_method,
      max_iter = max_iter
    )

  # Set integration to default in fdasrvf if not specified
  if (is.null(id)) {
    id = round(length(aligned$time)/2)
  }

  # Apply an elastic fPCA to the adjusted aligned training data using fdasrvf
  if (fpca_method == "jfpca") {
    fpca_res = fdasrvf::jointFPCA(
      warp_data = aligned,
      no = length(time),
      id = id,
      C = C,
      ci = ci,
      showplot = FALSE
    )
  } else if (fpca_method == "hfpca") {
    fpca_res = fdasrvf::horizFPCA(
      warp_data = aligned,
      no = length(time),
      ci = ci,
      showplot = FALSE
    )
  } else if (fpca_method == "vfpca") {
    fpca_res = fdasrvf::vertFPCA(
      warp_data = aligned,
      no = length(time),
      id = id,
      ci = ci,
      showplot = FALSE
    )
  } else {
    stop("fpca_method specified incorrectly. Must be 'jfpca', 'vfpca', or 'hfpca'.")
  }

  # Return the results from alignment and elastic fPCA
  list(alignment = aligned, fpca_type = fpca_method, fpca_res = fpca_res)

}
