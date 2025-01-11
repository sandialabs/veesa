library(testthat)
library(randomForest)
library(dplyr)

# Mock data for testing
set.seed(123)
mock_x <- data.frame(
  feature1 = rnorm(100),
  feature2 = rnorm(100),
  feature3 = rnorm(100)
)
mock_y <- factor(sample(c("A", "B"), 100, replace = TRUE))
mock_p <- factor(rbeta(n = 100, shape1 = 1, shape2 = 2))

# Train a random forest model
rf_model <- randomForest(
  formula = mock_y ~ .,
  data = mock_x
)

# Test 1: Check output structure
test_that("Output structure is correct", {
  result <- compute_pfi(
    x = mock_x,
    y = mock_y,
    f = rf_model,
    K = 5,
    metric = "accuracy"
  )
  expect_type(result, "list")
  expect_true("pfi" %in% names(result))
  expect_true("pfi_single_reps" %in% names(result))
})

# Test 2: Check dimensions of output
test_that("Output dimensions are correct", {
  result <- compute_pfi(
    x = mock_x,
    y = mock_y,
    f = rf_model,
    K = 5,
    metric = "accuracy"
  )
  
  expect_equal(length(result$pfi), ncol(mock_x))  # PFI should have length equal to number of features
  expect_equal(dim(result$pfi_single_reps), c(5, ncol(mock_x)))  # Single reps should be K x p
})

# Test 3: Check for error with mismatched dimensions
test_that("Function throws error with mismatched dimensions", {
  expect_error(compute_pfi(
    x = mock_x,
    y = mock_y[1:50],  # Mismatched length
    f = rf_model,
    K = 5,
    metric = "accuracy"
  ), "Number of observations in x and y disagree.")
})

# Test 4: Check for invalid metric
test_that("Function throws error with invalid metric", {
  expect_error(compute_pfi(
    x = mock_x,
    y = mock_y,
    f = rf_model,
    K = 5,
    metric = "invalid_metric"
  ), "'metric' specified incorrectly. Select either 'accuracy', 'logloss', or 'nmse'.")
})
