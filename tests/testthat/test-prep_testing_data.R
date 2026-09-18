library(testthat)
library(fdasrvf)

# Mock data for testing (100 time points, 10 functions)
set.seed(123)
mock_f <- matrix(rnorm(1000), nrow = 100, ncol = 10)
mock_time <- seq(0, 1, length.out = 100)

# Mock training preparation data
mock_train_prep <- prep_training_data(mock_f, mock_time, "jfpca")

# Test 1: Check output structure and dimensions
test_that("Output structure is correct", {
  result <- 
    prep_testing_data(
      f = mock_f, 
      time = mock_time, 
      train_prep = mock_train_prep, 
      optim_method = "DP"
    )
  expect_type(result, "list")
  expect_true(all(
    c("time", "f0", "fn", "q0", "qn", "mqn", "gam", "coef") %in% 
      names(result)
  ))
  expect_equal(dim(result$f0), dim(mock_f))
  expect_equal(dim(result$fn), dim(mock_f))
  expect_equal(dim(result$q0), dim(mock_f))
  expect_equal(dim(result$qn), dim(mock_f))
  expect_equal(length(result$mqn), nrow(mock_f))
  expect_equal(dim(result$coef), c(nrow(mock_f), ncol(mock_f)))
})

# Test 2: Check handling of different fpca types
test_that("Function handles different fpca types", {
  # hfPCA
  mock_train_prep = prep_training_data(mock_f, mock_time, "hfpca")
  result_hfpca <- 
    prep_testing_data(
      f = mock_f,
      time = mock_time, 
      train_prep = mock_train_prep, 
      optim_method = "DP"
    )
  expect_true("psi" %in% names(result_hfpca))
  expect_true("nu" %in% names(result_hfpca))
  # vfPCA
  mock_train_prep = prep_training_data(mock_f, mock_time, "vfpca")
  result_vfpca <- 
    prep_testing_data(
      f = mock_f, 
      time = mock_time, 
      train_prep = mock_train_prep, 
      optim_method = "DP"
    )
  expect_true("gam" %in% names(result_vfpca))
})

# Test 3: Check for error with incorrect input dimensions
test_that("Function throws error with incorrect input dimensions", {
  expect_error(
    prep_testing_data(
      f = matrix(1:20, nrow = 5), 
      time = mock_time, 
      train_prep = mock_train_prep, 
      optim_method = "DP"
    )
  )
})

# Test 4: Check for error with invalid optim_method
test_that("Function throws error with invalid optim_method", {
  expect_error(
    prep_testing_data(
      f = mock_f, 
      time = mock_time, 
      train_prep = mock_train_prep, 
      optim_method = "invalid_method"
    )
  )
})

# Test 5: Check that a test function is transformed the same way regardless of
#         which other test functions it is scored alongside
test_that("Coefficients do not depend on the rest of the test batch", {
  for (fpca_method in c("jfpca", "hfpca", "vfpca")) {
    train_prep <- prep_training_data(mock_f, mock_time, fpca_method)
    full <- prep_testing_data(mock_f, mock_time, train_prep, "DP")
    subset <- prep_testing_data(mock_f[, 1:3], mock_time, train_prep, "DP")
    single <- prep_testing_data(
      mock_f[, 1, drop = FALSE],
      mock_time,
      train_prep,
      "DP"
    )
    # Scoring a subset gives the same coefficients as scoring the full batch
    expect_equal(
      subset$coef,
      full$coef[, 1:3],
      tolerance = 1e-6,
      info = fpca_method
    )
    # A single test function is a valid batch and is not centered away
    expect_equal(ncol(single$coef), 1, info = fpca_method)
    expect_false(all(single$coef == 0), info = fpca_method)
    expect_equal(
      as.vector(single$coef),
      as.vector(full$coef[, 1]),
      tolerance = 1e-6,
      info = fpca_method
    )
  }
})

# Test 6: Check that the alignment penalty can be set explicitly
test_that("Function accepts an explicit alignment penalty", {
  result <- prep_testing_data(
    f = mock_f,
    time = mock_time,
    train_prep = mock_train_prep,
    optim_method = "DP",
    lambda = 0.01,
    penalty_method = "l2gam"
  )
  expect_type(result, "list")
  expect_equal(dim(result$coef), c(nrow(mock_f), ncol(mock_f)))
})

# Test 7: Check for correct output when using different optimization methods
test_that("Function works with different optimization methods", {
  result_dp <- 
    prep_testing_data(
      f = mock_f, 
      time = mock_time, 
      train_prep = mock_train_prep, 
      optim_method = "DPo"
    )
  result_dpo <- 
    prep_testing_data(
      f = mock_f, 
      time = mock_time,
      train_prep = mock_train_prep, 
      optim_method = "RBFGS"
    )
  expect_type(result_dp, "list")
  expect_type(result_dpo, "list")
})
