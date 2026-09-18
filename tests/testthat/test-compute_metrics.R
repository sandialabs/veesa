library(testthat)
library(randomForest)

# Mock data for testing
set.seed(123)
mock_x <- data.frame(
  feature1 = rnorm(100),
  feature2 = rnorm(100)
)
mock_y_class <- factor(sample(c("A", "B"), 100, replace = TRUE))
mock_y_num <- mock_x$feature1 * 2 + rnorm(100, sd = 0.1)

rf_class <- randomForest(x = mock_x, y = mock_y_class)
rf_reg <- randomForest(x = mock_x, y = mock_y_num)

# Test 1: Accuracy agrees with the proportion of correct predictions
test_that("compute_accuracy matches the proportion of correct predictions", {
  result <- compute_accuracy(mock_x, mock_y_class, rf_class, n = 100)
  expect_equal(result, mean(predict(rf_class, mock_x) == mock_y_class))
  expect_true(result >= 0 && result <= 1)
})

# Test 2: Log-loss works for responses whose levels are not numbers
test_that("compute_logloss handles non-numeric class labels", {
  result <- compute_logloss(mock_x, mock_y_class, rf_class, eps = 1e-15)
  expect_type(result, "double")
  expect_false(is.na(result))
  expect_true(result <= 0)
})

# Test 3: Log-loss agrees with the mean log-likelihood
test_that("compute_logloss matches the mean log-likelihood", {
  probs <- predict(rf_class, mock_x, type = "prob")
  p1 <- pmax(1e-15, pmin(1 - 1e-15, probs[, 1]))
  yind <- ifelse(as.character(mock_y_class) == colnames(probs)[1], 1, 0)
  expect_equal(
    compute_logloss(mock_x, mock_y_class, rf_class, eps = 1e-15),
    mean(yind * log(p1) + (1 - yind) * log(1 - p1))
  )
})

# Test 4: Log-loss rejects responses with more than two classes
test_that("compute_logloss errors for more than two classes", {
  y3 <- factor(sample(c("A", "B", "C"), 100, replace = TRUE))
  rf3 <- randomForest(x = mock_x, y = y3)
  expect_error(
    compute_logloss(mock_x, y3, rf3, eps = 1e-15),
    "'logloss' is only implemented for binary responses."
  )
})

# Test 5: Negative MSE agrees with the definition
test_that("compute_nmse matches the negative mean squared error", {
  result <- compute_nmse(mock_x, mock_y_num, rf_reg, n = 100)
  expect_equal(result, -mean((predict(rf_reg, mock_x) - mock_y_num)^2))
  expect_true(result <= 0)
})

# Test 6: PFI values are finite for each of the three metrics
test_that("compute_pfi returns finite values for each metric", {
  set.seed(1)
  expect_true(all(is.finite(
    compute_pfi(mock_x, mock_y_class, rf_class, K = 2, metric = "accuracy")$pfi
  )))
  expect_true(all(is.finite(
    compute_pfi(mock_x, mock_y_class, rf_class, K = 2, metric = "logloss")$pfi
  )))
  expect_true(all(is.finite(
    compute_pfi(mock_x, mock_y_num, rf_reg, K = 2, metric = "nmse")$pfi
  )))
})

# Test 7: Permuting a variable the model depends on lowers the metric
test_that("compute_pfi assigns a larger value to the informative variable", {
  set.seed(1)
  pfi <- compute_pfi(mock_x, mock_y_num, rf_reg, K = 5, metric = "nmse")$pfi
  expect_gt(pfi[1], pfi[2])
})
