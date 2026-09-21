library(testthat)

# These tests validate the packaged "shifted_peaks" dataset that is used
# throughout the examples and vignettes. They guard against the .rda file
# being regenerated with an unexpected structure. See
# https://github.com/sandialabs/veesa/blob/master/inst/shifted-peaks.md for
# the code used to prepare the dataset.

# Test 1: The dataset is a list with the three documented components
test_that("shifted_peaks has the documented top-level structure", {
  expect_type(shifted_peaks, "list")
  expect_named(
    shifted_peaks,
    c("data", "params", "true_means"),
    ignore.order = TRUE
  )
})

# Test 2: The 'data' component has the expected columns and factor variables
test_that("shifted_peaks$data has the expected columns", {
  expect_s3_class(shifted_peaks$data, "data.frame")
  expect_true(all(
    c("data", "id", "group", "index", "t", "y") %in% names(shifted_peaks$data)
  ))
  # 'data' splits the observations into training and testing
  expect_setequal(
    as.character(unique(shifted_peaks$data$data)),
    c("Training", "Testing")
  )
  # 'group' is a two-level factor
  expect_s3_class(shifted_peaks$data$group, "factor")
  expect_setequal(levels(shifted_peaks$data$group), c("1", "2"))
})

# Test 3: The data were simulated with 400 training and 100 testing functions
test_that("shifted_peaks$data has 400 training and 100 testing functions", {
  ids_by_split <-
    unique(shifted_peaks$data[, c("data", "id")])
  counts <- table(as.character(ids_by_split$data))
  expect_equal(unname(counts["Training"]), 400)
  expect_equal(unname(counts["Testing"]), 100)
  expect_equal(length(unique(shifted_peaks$data$id)), 500)
})

# Test 4: Each function is observed on the same time grid on [-15, 15]
test_that("shifted_peaks$data functions share a common time grid", {
  # Every function has the same number of samples
  samples_per_id <- table(shifted_peaks$data$id)
  expect_equal(length(unique(as.integer(samples_per_id))), 1)
  # Time runs over [-15, 15] (see inst/shifted-peaks.md: seq(-15, 15, ...))
  expect_gte(min(shifted_peaks$data$t), -15)
  expect_lte(max(shifted_peaks$data$t), 15)
  # There are no missing observations
  expect_false(any(is.na(shifted_peaks$data$y)))
})

# Test 5: The simulation parameters match the documented values
test_that("shifted_peaks$params holds the documented simulation parameters", {
  expect_type(shifted_peaks$params, "list")
  expect_equal(
    shifted_peaks$params,
    list(
      z1 = 1, z1_sd = 0.05, a1 = -3, a1_sd = 1,
      z2 = 1.25, z2_sd = 0.05, a2 = 3, a2_sd = 1
    )
  )
})

# Test 6: The true means describe both groups over the time grid
test_that("shifted_peaks$true_means describes both groups", {
  expect_s3_class(shifted_peaks$true_means, "data.frame")
  expect_true(all(
    c("group", "t", "mean_true") %in% names(shifted_peaks$true_means)
  ))
  expect_setequal(levels(shifted_peaks$true_means$group), c("1", "2"))
  expect_false(any(is.na(shifted_peaks$true_means$mean_true)))
})
