#' Simulate example functional data
#'
#' Function for simulating a set of functional data based on a deterministic function
#'     with covariates that affect the shape of the functions
#'
#' @param M Number of functions
#' @param N Number of samples per function
#' @param seed Seed for reproducibility
#'
#' @export simulate_functions
#'
#' @importFrom fdasrvf f_to_srvf jointFPCA optimum.reparam time_warping warp_f_gamma
#' @importFrom purrr map map2
#'
#' @returns Data frame with the following columns (where f is the function):
#' \itemize{
#'   \item t: "time" associated with sample from function where t in [0,1]
#'   \item y: f(t) for the particular observation
#'   \item x1: covariate 1 for function $f$ (constant across time)
#'   \item x2: covariate 2 for function $f$ (constant across time)
#'   \item x3: covariate 3 for function $f$ (constant across time)
#' }
#'
#' @details The functions are generated using the following equation:
#'
#' f(t) = (x_1*exp(-((t-0.3)^2)/0.005)) + (x_2(-((t-(0.7+x_3))^2/0.005)))
#'
#' where the covariates are generated as follows:
#'
#' \itemize{
#'   \item x_1 generated from Unif(0.1,1)
#'   \item x_2 generated from Unif(0.1,0.5)
#'   \item x_3 generated from Unif(-0.1,0.1)
#' }
#'
#' @examples
#' # Simulate data
#' sim_data = simulate_functions(M = 100, N = 75, seed = 20211130)

simulate_functions <- function(M, N, seed) {

  # Create a sequence of N sample times
  t = seq(0, 1, length.out = N)

  # Generate pairs of independent covariate values
  # based on the number functions specified (M)
  x1 = runif(n = M, min = 0.1, max = 1)
  x2 = runif(n = M, min = 0.1, max = 0.5)
  x3 = runif(n = M, min = -0.1, max = 0.1)

  # Generate the functions and store in a data frame
  set.seed(seed)
  res = purrr::pmap_df(
    .l = list(x1, x2, x3),
    .f = function(x1, x2, x3, t) {
      data.frame(
        t = t,
        y = (x1 * exp(-(t - 0.3)^2 / 0.005)) + (x2 * exp(-(t - (0.7 + x3))^2 / 0.005)),
        x1 = x1,
        x2 = x2,
        x3 = x3
      )
    },
    t = t,
    .id = "id"
  )

  # Return the simulated functions
  return(res)

}
