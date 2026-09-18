library(testthat)
library(fdasrvf)

# Mock data for testing (50 time points, 10 functions)
set.seed(123)
mock_f <- matrix(rnorm(500), nrow = 50, ncol = 10)
mock_time <- seq(0, 1, length.out = 50)

# Test 1: Check that the unimplemented fpca types are rejected
test_that("Function throws errors for fpca types that are not supported", {
  expect_error(
    align_pcdirs(list(fpca_type = "hfpca")),
    "Alignment of principal directions is not currently implemented."
  )
  expect_error(
    align_pcdirs(list(fpca_type = "vfpca")),
    "Alignment of principal directions is not necessary for vertical fPCA."
  )
})

# Test 2: Check output structure
test_that("Output has the same structure as the training object", {
  mock_train <- prep_training_data(
    mock_f,
    mock_time,
    "jfpca",
    center_warpings = FALSE
  )
  result <- align_pcdirs(mock_train)
  expect_named(result, names(mock_train))
  expect_equal(result$fpca_type, "jfpca")
  expect_equal(dim(result$fpca_res$f_pca), dim(mock_train$fpca_res$f_pca))
  expect_false(any(is.na(result$fpca_res$f_pca)))
})

# Test 3: Check that gamI is returned alongside the aligned directions
test_that("gamI is added to the fpca_res object", {
  mock_train <- prep_training_data(
    mock_f,
    mock_time,
    "jfpca",
    center_warpings = FALSE
  )
  result <- align_pcdirs(mock_train)
  expect_true("gamI" %in% names(result$fpca_res))
  expect_equal(length(result$fpca_res$gamI), nrow(mock_f))
  expect_false(any(is.na(result$fpca_res$gamI)))
})
