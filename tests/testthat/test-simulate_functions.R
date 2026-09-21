library(testthat)

# Test 1: Check output structure
test_that("Output structure is correct", {
  result <- simulate_functions(M = 5, N = 100, seed = 123)
  expect_equal(ncol(result), 6)  # Should have 5 columns: id, t, y, x1, x2, x3
  expect_true(all(c("id", "t", "y", "x1", "x2", "x3") %in% names(result)))
})

# Test 2: Check number of functions generated
test_that("Correct number of functions generated", {
  M <- 10
  N <- 50
  result <- simulate_functions(M = M, N = N, seed = 123)
  expect_equal(length(unique(result$id)), M)  # Unique IDs should match M
  expect_equal(nrow(result), M * N)  # Total rows should be M * N
})

# Test 3: Check time variable
test_that("Time variable is correct", {
  N <- 100
  result <- simulate_functions(M = 1, N = N, seed = 123)
  expect_equal(result$t, seq(0, 1, length.out = N))  # Time should be from 0 to 1
})

# Test 4: Check covariate ranges
test_that("Covariate ranges are correct", {
  result <- simulate_functions(M = 5, N = 100, seed = 123)
  expect_true(all(result$x1 >= 0.1 & result$x1 <= 1))  # x1 should be in [0.1, 1]
  expect_true(all(result$x2 >= 0.1 & result$x2 <= 0.5))  # x2 should be in [0.1, 0.5]
  expect_true(all(result$x3 >= -0.1 & result$x3 <= 0.1))  # x3 should be in [-0.1, 0.1]
})

# Test 5: Check reproducibility with seed
test_that("Function is reproducible with the same seed", {
  result1 <- simulate_functions(M = 5, N = 100, seed = 123)
  result2 <- simulate_functions(M = 5, N = 100, seed = 123)
  expect_equal(result1, result2)  # Outputs should be identical
})

# Test 6: Check output for different seeds
test_that("Function produces different outputs with different seeds", {
  result1 <- simulate_functions(M = 1, N = 100, seed = 123)
  result2 <- simulate_functions(M = 1, N = 100, seed = 456)
  expect_false(identical(result1$y, result2$y))  # Outputs should not be identical
})

# Test 7: Check that the setting of the seed does not depend on the global RNG
#         state (the function calls set.seed internally)
test_that("Function output does not depend on the ambient RNG state", {
  set.seed(1)
  result1 <- simulate_functions(M = 3, N = 50, seed = 999)
  set.seed(2)
  runif(10)
  result2 <- simulate_functions(M = 3, N = 50, seed = 999)
  expect_equal(result1, result2)
})

# Test 8: Check that y is computed from the documented deterministic equation
test_that("y matches the deterministic generating equation", {
  M <- 4
  N <- 60
  result <- simulate_functions(M = M, N = N, seed = 20211130)
  t <- seq(0, 1, length.out = N)
  # Recompute y for each function from its (constant) covariates
  for (fun_id in unique(result$id)) {
    sub <- result[result$id == fun_id, ]
    x1 <- unique(sub$x1)
    x2 <- unique(sub$x2)
    x3 <- unique(sub$x3)
    expect_length(x1, 1)
    expected_y <-
      (x1 * exp(-(t - 0.3)^2 / 0.005)) +
      (x2 * exp(-(t - (0.7 + x3))^2 / 0.005))
    expect_equal(sub$y, expected_y)
  }
})

# Test 9: Check the types of the returned columns
test_that("Returned columns have the expected types", {
  result <- simulate_functions(M = 2, N = 30, seed = 7)
  expect_type(result$id, "character")
  expect_type(result$t, "double")
  expect_type(result$y, "double")
  expect_false(any(is.na(result$y)))
})

