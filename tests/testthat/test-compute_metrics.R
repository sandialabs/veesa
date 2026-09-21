library(testthat)

# 'randomForest' is a Suggests dependency, so skip the whole file when it is not
# installed rather than failing at load time.
skip_if_not_installed("randomForest")
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

# Test 8: Accuracy divides the number of correct predictions by n, so passing a
#         larger n rescales the result accordingly
test_that("compute_accuracy scales by the supplied n", {
  correct <- sum(predict(rf_class, mock_x) == mock_y_class)
  expect_equal(compute_accuracy(mock_x, mock_y_class, rf_class, n = 100), correct / 100)
  # Doubling n halves the reported value (n is used verbatim as the denominator)
  expect_equal(
    compute_accuracy(mock_x, mock_y_class, rf_class, n = 200),
    correct / 200
  )
})

# Test 9: Log-loss clips probabilities to [eps, 1 - eps] so it stays finite even
#         when a class probability is exactly 0 or 1
test_that("compute_logloss clips extreme probabilities via eps", {
  # A perfectly separable problem drives predicted probabilities to 0 / 1
  set.seed(42)
  sep_x <- data.frame(feature1 = c(rnorm(50, -5), rnorm(50, 5)))
  sep_y <- factor(rep(c("A", "B"), each = 50))
  sep_rf <- randomForest(x = sep_x, y = sep_y)
  result <- compute_logloss(sep_x, sep_y, sep_rf, eps = 1e-15)
  expect_true(is.finite(result))
  # A larger eps clips more aggressively, giving a value no smaller (closer to 0)
  result_loose <- compute_logloss(sep_x, sep_y, sep_rf, eps = 1e-3)
  expect_gte(result_loose, result)
})

# Test 10: The multivariate branch of compute_nmse (used for randomForestSRC
#          multivariate models) treats the responses as a single stacked vector.
#          A lightweight fake model with a predict method exercises this branch
#          without depending on randomForestSRC.
test_that("compute_nmse handles matrix responses (multivariate branch)", {
  # Build a fake model whose predictions we control
  fake_model <- structure(
    list(preds = list(r1 = c(1, 2, 3), r2 = c(4, 5, 6))),
    class = "veesa_fake_mv"
  )
  # predict returns the $regrOutput structure the multivariate branch expects
  predict.veesa_fake_mv <<- function(object, newdata, ...) {
    list(regrOutput = list(
      r1 = list(predicted = object$preds$r1),
      r2 = list(predicted = object$preds$r2)
    ))
  }
  on.exit(rm(predict.veesa_fake_mv, envir = globalenv()), add = TRUE)

  x_mv <- data.frame(a = 1:3, b = 4:6)
  y_mv <- matrix(c(1.5, 2, 2.5, 4, 5.5, 6), nrow = 3, ncol = 2)

  result <- compute_nmse(x_mv, y_mv, fake_model, n = 3)

  # Expected: stack predictions (r1 then r2) and the response as a vector, then
  # take the negative mean squared error over all stacked elements
  yhat <- c(1, 2, 3, 4, 5, 6)
  yvec <- as.vector(y_mv)
  expected <- -sum((yhat - yvec)^2) / length(yhat)
  expect_equal(result, expected)
  expect_true(result <= 0)
})

# Test 11: compute_pfi reports one PFI per feature and one row per repetition,
#          and the averaged PFI is the row-mean of the single-rep matrix
test_that("compute_pfi averages the single-repetition importances", {
  set.seed(3)
  result <- compute_pfi(mock_x, mock_y_num, rf_reg, K = 4, metric = "nmse")
  expect_equal(length(result$pfi), ncol(mock_x))
  expect_equal(dim(result$pfi_single_reps), c(4, ncol(mock_x)))
  expect_equal(result$pfi, colMeans(result$pfi_single_reps))
})
