# library(testthat)
# library(ggplot2)
# library(dplyr)
# library(tidyr)
# library(purrr)
# 
# # Mock data for testing
# set.seed(123)
# mock_f <- matrix(rnorm(1000), nrow = 100, ncol = 10)  # 10 time points, 10 functions
# mock_time <- seq(0, 1, length.out = 100)
# 
# # Mock training preparation data
# mock_fd <- prep_training_data(mock_f, mock_time, "jfpca")
# 
# # Test 1: Check output structure
# test_that("Output is a ggplot object", {
#   result <- plot_pc_directions(
#     fpcs = 1,
#     fdasrvf = mock_fd$fpca_res,
#     fpca_method = "jfpca"
#   )
#   expect_s3_class(result, "ggplot")
# })
# 
# # Test 2: Check error for invalid fpca_method
# test_that("Function throws error with invalid fpca_method", {
#   expect_error(plot_pc_directions(
#     fpcs = 1,
#     fdasrvf = mock_fd,
#     fpca_method = "invalid_method"
#   ), "'fpca_method' entered incorrectly. Must be 'jfpca', 'vfpca', or 'hfpca'.")
# })
# 
# # Test 3: Check handling of multiple PCs
# test_that("Function handles multiple PCs correctly", {
#   result <- plot_pc_directions(
#     fpcs = 1:3,
#     fdasrvf = mock_fd$fpca_res,
#     fpca_method = "jfpca"
#   )
#   expect_s3_class(result, "ggplot")
# })
# 
# # Test 4: Check for correct number of lines in the plot
# test_that("Plot contains correct number of lines", {
#   result <- plot_pc_directions(
#     fpcs = 1,
#     fdasrvf = mock_fd$fpca_res,
#     fpca_method = "jfpca"
#   )
#   # Check the number of lines in the plot
#   expect_equal(length(result$layers), 1)  # Should have one layer for the line
# })
# 
# # Test 5: Check for correct alpha values
# test_that("Function applies correct alpha values", {
#   result <- plot_pc_directions(
#     fpcs = 1,
#     fdasrvf = mock_fd$fpca_res,
#     fpca_method = "jfpca",
#     alpha = 0.5
#   )
#   expect_equal(result$layers[[1]]$aes_params$alpha, 0.5)  # Check if alpha is set correctly
# })
