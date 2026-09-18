library(testthat)
library(ggplot2)

# The plot functions only need the pieces of an elastic fPCA object that they
# read, so a small mock object is used to keep the tests independent of the
# alignment routines in fdasrvf.
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
  result <- plot_pc_directions(
    fpcs = 1,
    fdasrvf = mock_jfpca,
    fpca_method = "jfpca"
  )
  expect_s3_class(result, "ggplot")
  expect_no_warning(ggplot_build(result))
})

# Test 2: Check error for invalid fpca_method
test_that("Function throws error with invalid fpca_method", {
  expect_error(plot_pc_directions(
    fpcs = 1,
    fdasrvf = mock_jfpca,
    fpca_method = "invalid_method"
  ), "'fpca_method' entered incorrectly. Must be 'jfpca', 'vfpca', or 'hfpca'.")
})

# Test 3: Check handling of multiple PCs
test_that("Function handles multiple PCs correctly", {
  result <- plot_pc_directions(
    fpcs = 1:3,
    fdasrvf = mock_jfpca,
    fpca_method = "jfpca"
  )
  expect_s3_class(result, "ggplot")
  expect_equal(nlevels(result$data$fpc_facet), 3)
})

# Test 4: Check for correct number of lines in the plot
test_that("Plot contains correct number of lines", {
  result <- plot_pc_directions(
    fpcs = 1,
    fdasrvf = mock_jfpca,
    fpca_method = "jfpca"
  )
  # Check the number of lines in the plot
  expect_equal(length(result$layers), 1)  # Should have one layer for the line
})

# Test 5: Check for correct alpha values
test_that("Function applies correct alpha values", {
  result <- plot_pc_directions(
    fpcs = 1,
    fdasrvf = mock_jfpca,
    fpca_method = "jfpca",
    alpha = 0.5
  )
  expect_equal(result$layers[[1]]$aes_params$alpha, 0.5)  # Check if alpha is set correctly
})

# Test 6: Check that hfpca objects are handled
test_that("Function handles hfpca objects", {
  result <- plot_pc_directions(
    fpcs = 1:2,
    fdasrvf = mock_hfpca,
    fpca_method = "hfpca"
  )
  expect_s3_class(result, "ggplot")
  expect_no_warning(ggplot_build(result))
})

# Test 7: Check the proportion of variability reported in the facet labels
test_that("Proportion of variability explained is computed from the eigenvalues", {
  result <- plot_pc_directions(
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

# Test 8: Check that the full spectrum is used as the denominator when the
#         fPCA object retains only some of the principal components
test_that("Retained PCs are scaled by the full spectrum when it is available", {
  truncated <- mock_jfpca
  truncated$latent <- mock_jfpca$latent[1:2]
  truncated$f_pca <- mock_jfpca$f_pca[, , 1:2]
  result <- plot_pc_directions(
    fpcs = 1:2,
    fdasrvf = truncated,
    fpca_method = "jfpca",
    digits = 2
  )
  expect_equal(
    levels(result$data$fpc_facet),
    c("jfPC 1 (55.56%)", "jfPC 2 (27.78%)")
  )
})

# Test 9: Check the formatting of very small percentages
test_that("PCs that round to zero are reported as smaller than the resolution", {
  small <- list(
    latent = c(1000, 4),
    time = seq(0, 1, length.out = mock_M),
    f_pca = array(rnorm(mock_M * 5 * 2), dim = c(mock_M, 5, 2))
  )
  result <- plot_pc_directions(1:2, small, "jfpca", digits = 0)
  expect_equal(levels(result$data$fpc_facet), c("jfPC 1 (100%)", "jfPC 2 (<1%)"))
  result <- plot_pc_directions(1:2, small, "jfpca", digits = 2)
  expect_equal(levels(result$data$fpc_facet), c("jfPC 1 (99.6%)", "jfPC 2 (0.4%)"))
})

# Test 10: Check validation of the scalar plotting arguments
test_that("Function throws errors for invalid arguments", {
  expect_error(
    plot_pc_directions(1, mock_jfpca, "jfpca", linetype = c("dashed", "solid")),
    "'linetype' must be a single logical value."
  )
  expect_error(
    plot_pc_directions(1, mock_jfpca, "jfpca", alpha = c(0.2, 1, 0.2)),
    "'alpha' must be a single value."
  )
  expect_error(
    plot_pc_directions(1:9, mock_jfpca, "jfpca"),
    "'fpcs' contains principal components that are not in the fPCA object."
  )
})
