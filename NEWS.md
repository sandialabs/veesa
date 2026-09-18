# veesa 0.1.8

- Fixed the proportion of variability reported by `plot_pc_directions` and
  `plot_pc_diffs`: `latent` from fdasrvf already holds the eigenvalues of the
  covariance matrix, so the values are no longer squared before being scaled
- Fixed `compute_logloss` (and therefore `compute_pfi` with `metric = "logloss"`)
  returning `NA` when the response levels are not numbers, and added an error
  for responses with more than two classes
- Fixed `prep_testing_data` centering the hfPCA and vfPCA coefficients with
  statistics computed from the test data instead of the training data, which
  made a test function's coefficients depend on the rest of the test batch
- `prep_testing_data` now aligns the test data with the same elasticity and
  penalty that were used on the training data, and accepts `lambda` and
  `penalty_method` to override them. Both are validated, `"norm"` is accepted
  as an alias for `"l2gam"` (`fdasrvf::optimum.reparam` only takes the current
  penalty names), and the settings used are returned in a new `call` component
- `prep_training_data` now validates `lambda` and `penalty_method` before
  passing them to `fdasrvf::time_warping`, and the documentation of `lambda`
  describes its role as the weight on the alignment penalty
- Fixed `center_warping_funs` leaving the SRSF mean uncentered
- Exported `align_pcdirs` and `center_warping_funs`, which were documented but
  unavailable
- Replaced the deprecated `size` aesthetic and tidyselect usage in the plotting
  functions, which removes the deprecation warnings raised on every call
- Corrected the documentation of `alpha` and `linetype` in the plotting
  functions, which described them as vectors when they are single values
- Added tests for the plotting functions, the metric helpers, `align_pcdirs`,
  and `center_warping_funs`

# veesa 0.1.7

- Added a function for plotting the differences between the PC directions and the mean

# veesa 0.1.6

- Resubmission to CRAN with more fixes
- Made it to CRAN

# veesa 0.1.5

- Resubmission to CRAN with fixes

# veesa 0.1.4

- Initial CRAN submission!
- Documentation clean up

# veesa 0.1.3

- Updated inputs in prep_training_data to match fdasrvf
- Added examples to documentation
- Cleaned up wording in documentation a bit
- Added tests

# veesa 0.1.2

- Added shifted peaks data
- Added example code from manuscript

# veesa 0.1.1

- Adjusted title in plot_pc_directions
- Updated code to new version of fdasrvf

# veesa 0.1.0

- Initial version of package
