#' Obtain PC directions with centered warping functions
#'
#' The function 'prep_training_data' does not center the warping functions,
#' which leads to issues when visualizing joint and horizontal principal
#' component directions. This function aligns the principal directions for
#' improved interpretability of the principal directions. Currently, only
#' alignment for jfPCA has been implemented.
#'
#' @param train_obj Output object from `prep_training_data` (jfpca only)
#'
#' @return List with the same structure as `prep_training_data`, but
#' the principal directions are replaced with the aligned version and gamI is
#' included in the fpca_res object.

align_pcdirs <- function(train_obj) {

  if (train_obj$fpca_type == "hfpca") {
    stop("Alignment of principal directions is not currently implemented.")
  } else if (train_obj$fpca_type == "vfpca") {
    stop("Alignment of principal directions is not necessary for vertical fPCA.")
  }

  # Extract and prepare objects based on training data
  aligned = train_obj$alignment
  gam = t(aligned$warping_functions)
  N = nrow(gam)
  M = ncol(gam)
  mq = aligned$mqn
  pc_dirs = train_obj$fpca_res$f_pca

  # Compute gamma inverse
  gamI = fdasrvf::SqrtMeanInverse(t(gam))

  # Extract number of principal directions and number of PCs
  ndirs = dim(pc_dirs)[2]
  npcs = dim(pc_dirs)[3]

  # Create empty array to store aligned principal directions
  aligned_pcdirs = array(data = NA, dim = dim(pc_dirs))

  # Align PC directions
  for(pc in 1:npcs) {
    for (dir in 1:ndirs) {
      aligned_pcdirs[, dir, pc] <-
        fdasrvf::warp_f_gamma(
          f = pc_dirs[, dir, pc],
          time = aligned$time,
          gamma = gamI
        )
    }
  }

  # Replace principal directions with the aligned versions
  train_obj$fpca_res$f_pca = aligned_pcdirs

  # Add gamI to train object
  train_obj$fpca_res$gamI = gamI

  # Return the updated train object
  return(train_obj)

}
