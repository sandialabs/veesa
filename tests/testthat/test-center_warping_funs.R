library(testthat)
library(fdasrvf)

# Mock data for testing (50 time points, 10 functions)
set.seed(123)
mock_f <- matrix(rnorm(500), nrow = 50, ncol = 10)
mock_time <- seq(0, 1, length.out = 50)

# The warping functions are left uncentered by time_warping so that centering
# them here does something other than reproduce the input
mock_train <- prep_training_data(
  mock_f,
  mock_time,
  "jfpca",
  center_warpings = FALSE
)

# Test 1: Check output structure
test_that("Output has the same structure as the training object", {
  result <- center_warping_funs(mock_train)
  expect_named(result, names(mock_train))
  expect_named(result$alignment, names(mock_train$alignment))
  expect_equal(dim(result$alignment$qn), dim(mock_train$alignment$qn))
  expect_equal(dim(result$alignment$fn), dim(mock_train$alignment$fn))
  expect_equal(
    dim(result$alignment$warping_functions),
    dim(mock_train$alignment$warping_functions)
  )
  expect_equal(length(result$alignment$mqn), length(mock_train$alignment$mqn))
})

# Test 2: Check that the SRSF mean is centered along with the aligned functions
test_that("The SRSF mean is centered", {
  result <- center_warping_funs(mock_train)
  time <- mock_train$alignment$time
  gam <- mock_train$alignment$warping_functions
  M <- nrow(gam)
  gamI <- fdasrvf::SqrtMeanInverse(gam)
  gamI_dev <- fdasrvf::gradient(gamI, 1 / (M - 1))
  time0 <- (time[length(time)] - time[1]) * gamI + time[1]
  expected_mqn <-
    stats::approx(time, mock_train$alignment$mqn, xout = time0)$y *
    sqrt(gamI_dev)
  expect_equal(result$alignment$mqn, expected_mqn)
  expect_false(any(is.na(result$alignment$mqn)))
})

# Test 3: Check that the centered results contain no missing values
test_that("Centered functions contain no missing values", {
  result <- center_warping_funs(mock_train)
  expect_false(any(is.na(result$alignment$qn)))
  expect_false(any(is.na(result$alignment$fn)))
  expect_false(any(is.na(result$alignment$warping_functions)))
})
