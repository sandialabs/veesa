library(testthat)
library(fdasrvf)

# Mock data for testing
set.seed(123)
mock_f <- matrix(rnorm(1000), nrow = 100, ncol = 10)  # 10 time points, 10 functions
mock_time <- seq(0, 1, length.out = 100)

# Test 1: Check output structure and dimensions
test_that("Output structure is correct", {
  result <- prep_training_data(mock_f, mock_time, "jfpca")
  expect_type(result, "list")
  expect_true(all(c("alignment", "fpca_type", "fpca_res") %in% names(result)))
  expect_equal(dim(result$alignment$f0), dim(mock_f))
  expect_equal(nrow(result$fpca_res$f_pca[,,1]), nrow(mock_f))
})

# Test 2: Check handling of different fpca methods
test_that("Function handles different fpca methods", {
  result_jfpca <- prep_training_data(mock_f, mock_time, fpca_method = "jfpca")
  result_hfpca <- prep_training_data(mock_f, mock_time, fpca_method = "hfpca")
  result_vfpca <- prep_training_data(mock_f, mock_time, fpca_method = "vfpca")
  expect_equal(result_jfpca$fpca_type, "jfpca")
  expect_equal(result_hfpca$fpca_type, "hfpca")
  expect_equal(result_vfpca$fpca_type, "vfpca")
})

# Test 3: Check for error with incorrect fpca_method
test_that("Function throws error with incorrect fpca_method", {
  expect_error(
    prep_training_data(mock_f, mock_time, fpca_method = "invalid_method"), 
    "fpca_method specified incorrectly. Must be 'jfpca', 'vfpca', or 'hfpca'."
  )
})

# Test 4: Check for error with incorrect input dimensions
test_that("Function throws error with incorrect input dimensions", {
  expect_error(
    prep_training_data(
      f = matrix(1:100, nrow = 5), 
      time = mock_time,
      fpca_method = "jfpca"
    )
  )
})

# Test 5: Check default values for optional parameters
test_that("Default values for optional parameters are set correctly", {
  result <- prep_training_data(mock_f, mock_time, fpca_method = "jfpca")
  expect_equal(result$alignment$call$lambda, 0) 
  expect_equal(result$alignment$call$penalty_method, "roughness")
  expect_equal(result$alignment$call$centroid_type, "mean")
  expect_true(result$alignment$call$center_warpings)
})

# # Test 6: Check parallel processing option
# test_that("Function runs with parallel processing", {
#   result <-
#     prep_training_data(
#       f = mock_f, 
#       time = mock_time,
#       fpca_method = "jfpca", 
#       parallel = TRUE
#     )
#   expect_type(result, "list")
# })
