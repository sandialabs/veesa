library(testthat)
library(ggplot2)

# As in test-plot_pc_directions.R, a mock elastic fPCA object is used so that
# the tests do not depend on the alignment routines in fdasrvf.
set.seed(123)
mock_M <- 25
mock_jfpca <- list(
  latent = c(10, 5, 2, 1),
  eigs = c(10, 5, 2, 1),
  time = seq(0, 1, length.out = mock_M),
  f_pca = array(rnorm(mock_M * 5 * 4), dim = c(mock_M, 5, 4))
)
mock_hfpca <- list(
  latent = c(10, 5, 2, 1),
  gam_pca = array(rnorm(5 * mock_M * 4), dim = c(5, mock_M, 4))
)

# Test 1: Check output structure
test_that("Output is a ggplot object", {
  result <- plot_pc_diffs(
    fpcs = 1,
    fdasrvf = mock_jfpca,
    fpca_method = "jfpca"
  )
  expect_s3_class(result, "ggplot")
  expect_no_warning(ggplot_build(result))
})

# Test 2: Check error for invalid fpca_method
test_that("Function throws error with invalid fpca_method", {
  expect_error(plot_pc_diffs(
    fpcs = 1,
    fdasrvf = mock_jfpca,
    fpca_method = "invalid_method"
  ), "'fpca_method' entered incorrectly. Must be 'jfpca', 'vfpca', or 'hfpca'.")
})

# Test 3: Check handling of multiple PCs and fpca types
test_that("Function handles multiple PCs and hfpca objects", {
  result <- plot_pc_diffs(fpcs = 1:3, fdasrvf = mock_jfpca, fpca_method = "jfpca")
  expect_s3_class(result, "ggplot")
  expect_equal(nlevels(result$data$fpc_facet), 3)
  result_hfpca <- plot_pc_diffs(fpcs = 1:2, fdasrvf = mock_hfpca, fpca_method = "hfpca")
  expect_s3_class(result_hfpca, "ggplot")
  expect_no_warning(ggplot_build(result_hfpca))
})

# Test 4: Check that the Karcher mean is differenced out rather than plotted
test_that("Lines are the differences from the Karcher mean", {
  result <- plot_pc_diffs(fpcs = 1, fdasrvf = mock_jfpca, fpca_method = "jfpca")
  expect_equal(levels(result$data$line), c("-2SD", "-1SD", "+1SD", "+2SD"))
  expect_true("diff" %in% names(result$data))
  expect_equal(
    result$data$diff,
    result$data$`Karcher Mean` - result$data$value
  )
})

# Test 5: Check the proportion of variability reported in the facet labels
test_that("Proportion of variability explained is computed from the eigenvalues", {
  result <- plot_pc_diffs(
    fpcs = 1:4,
    fdasrvf = mock_jfpca,
    fpca_method = "jfpca",
    digits = 2
  )
  expect_equal(
    levels(result$data$fpc_facet),
    c("jfPC 1 (55.56%)", "jfPC 2 (27.78%)", "jfPC 3 (11.11%)", "jfPC 4 (5.56%)")
  )
})

# Test 6: Check validation of the scalar plotting arguments
test_that("Function throws errors for invalid arguments", {
  expect_error(
    plot_pc_diffs(1, mock_jfpca, "jfpca", linetype = c("dashed", "solid")),
    "'linetype' must be a single logical value."
  )
  expect_error(
    plot_pc_diffs(1, mock_jfpca, "jfpca", alpha = c(0.2, 1, 0.2)),
    "'alpha' must be a single value."
  )
  expect_error(
    plot_pc_diffs(1:9, mock_jfpca, "jfpca"),
    "'fpcs' contains principal components that are not in the fPCA object."
  )
})

# Test 7: Check that the plot can be built without line types
test_that("Function works with linetype turned off", {
  result <- plot_pc_diffs(1:2, mock_jfpca, "jfpca", linetype = FALSE)
  expect_s3_class(result, "ggplot")
  expect_no_warning(ggplot_build(result))
})

# Test 8: Check that vfpca objects are handled
test_that("Function handles vfpca objects", {
  mock_vfpca <- list(
    latent = c(10, 5, 2, 1),
    time = seq(0, 1, length.out = mock_M),
    f_pca = array(rnorm(mock_M * 5 * 4), dim = c(mock_M, 5, 4))
  )
  result <- plot_pc_diffs(1:2, mock_vfpca, "vfpca")
  expect_s3_class(result, "ggplot")
  expect_no_warning(ggplot_build(result))
  expect_equal(nlevels(result$data$fpc_facet), 2)
})

# Test 9: Check that an explicit time vector and free y-axis scales are accepted
test_that("Function accepts explicit times and freey", {
  result <- plot_pc_diffs(
    fpcs = 1:2,
    fdasrvf = mock_jfpca,
    fpca_method = "jfpca",
    times = seq(-1, 1, length.out = mock_M),
    freey = TRUE
  )
  expect_s3_class(result, "ggplot")
  expect_no_warning(ggplot_build(result))
})
