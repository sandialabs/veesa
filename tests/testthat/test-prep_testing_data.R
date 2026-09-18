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

# Test 5: Check for correct output when using different optimization methods
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

# Test 6: Check that the alignment penalty is inherited from the training data
test_that("Alignment penalty is inherited from the training data", {
  result <-
    prep_testing_data(
      f = mock_f,
      time = mock_time,
      train_prep = mock_train_prep,
      optim_method = "DP"
    )
  expect_equal(result$call$lambda, mock_train_prep$alignment$call$lambda)
  expect_equal(
    result$call$penalty_method,
    mock_train_prep$alignment$call$penalty_method
  )
  # Training data used the defaults (no penalty)
  expect_equal(result$call$lambda, 0)
  expect_equal(result$call$penalty_method, "roughness")
})

# Test 7: Check that the alignment penalty can be overridden
test_that("Alignment penalty can be overridden", {
  result <-
    prep_testing_data(
      f = mock_f,
      time = mock_time,
      train_prep = mock_train_prep,
      optim_method = "DP",
      lambda = 0.01,
      penalty_method = "l2gam"
    )
  expect_equal(result$call$lambda, 0.01)
  expect_equal(result$call$penalty_method, "l2gam")
  expect_equal(dim(result$fn), dim(mock_f))
})

# Test 8: Check that "norm" is treated as an alias for "l2gam"
test_that("Penalty alias 'norm' is converted to 'l2gam'", {
  result <-
    prep_testing_data(
      f = mock_f,
      time = mock_time,
      train_prep = mock_train_prep,
      optim_method = "DP",
      lambda = 0.01,
      penalty_method = "norm"
    )
  expect_equal(result$call$penalty_method, "l2gam")
})

# Test 9: Check for errors with invalid penalty specifications
test_that("Function throws error with invalid penalty specifications", {
  expect_error(
    prep_testing_data(
      f = mock_f,
      time = mock_time,
      train_prep = mock_train_prep,
      penalty_method = "invalid_penalty"
    )
  )
  expect_error(
    prep_testing_data(
      f = mock_f,
      time = mock_time,
      train_prep = mock_train_prep,
      lambda = "a"
    ),
    "lambda must be a single numeric value."
  )
})
