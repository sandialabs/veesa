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

# Test 4: Check that only the principal directions are altered; the alignment
#         output and fpca_type are passed through untouched
test_that("Alignment output and fpca_type are left unchanged", {
  mock_train <- prep_training_data(
    mock_f,
    mock_time,
    "jfpca",
    center_warpings = FALSE
  )
  result <- align_pcdirs(mock_train)
  expect_identical(result$alignment, mock_train$alignment)
  expect_identical(result$fpca_type, mock_train$fpca_type)
  # Every element of fpca_res other than the (aligned) directions and the new
  # gamI is carried over unchanged
  carried <- setdiff(names(mock_train$fpca_res), c("f_pca", "gamI"))
  for (nm in carried) {
    expect_identical(result$fpca_res[[nm]], mock_train$fpca_res[[nm]], info = nm)
  }
})

# Test 5: gamI is a valid warping function on [0, 1] (monotone, spanning the
#         unit interval), matching what fdasrvf::SqrtMeanInverse returns
test_that("gamI is a valid warping function", {
  mock_train <- prep_training_data(
    mock_f,
    mock_time,
    "jfpca",
    center_warpings = FALSE
  )
  gamI <- align_pcdirs(mock_train)$fpca_res$gamI
  expect_equal(gamI[1], 0, tolerance = 1e-6)
  expect_equal(gamI[length(gamI)], 1, tolerance = 1e-6)
  expect_true(all(diff(gamI) >= -1e-8))  # non-decreasing
})

# Test 6: The aligned directions are genuinely warped (they differ from the
#         originals whenever gamI is not the identity), while their range stays
#         comparable to the input
test_that("Aligned directions differ from the originals under non-trivial warping", {
  mock_train <- prep_training_data(
    mock_f,
    mock_time,
    "jfpca",
    center_warpings = FALSE
  )
  original <- mock_train$fpca_res$f_pca
  aligned <- align_pcdirs(mock_train)$fpca_res$f_pca
  gamI <- align_pcdirs(mock_train)$fpca_res$gamI
  # gamI departs from the identity, so the directions should change
  identity_gam <- seq(0, 1, length.out = length(gamI))
  if (max(abs(gamI - identity_gam)) > 1e-6) {
    expect_false(isTRUE(all.equal(aligned, original)))
  }
  expect_equal(dim(aligned), dim(original))
  expect_false(any(is.na(aligned)))
})
