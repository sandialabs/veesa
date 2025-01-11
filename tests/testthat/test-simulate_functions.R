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

# # Test 5: Check reproducibility with seed
# test_that("Function is reproducible with the same seed", {
#   result1 <- simulate_functions(M = 5, N = 100, seed = 123)
#   result2 <- simulate_functions(M = 5, N = 100, seed = 123)
#   expect_equal(result1$y, result2$y)  # Outputs should be identical
# })
# 
# # Test 6: Check output for different seeds
# test_that("Function produces different outputs with different seeds", {
#   result1 <- simulate_functions(M = 1, N = 100, seed = 123)
#   result2 <- simulate_functions(M = 1, N = 100, seed = 456)
#   expect_false(identical(result1$y, result2$y))  # Outputs should not be identical
# })

