# veesa (development version)

- Exposed the alignment penalty in `prep_testing_data` via the new `lambda` and
  `penalty_method` arguments. By default, both are inherited from the training
  data alignment, so the testing data are now aligned under the same criterion
  as the training data (previously the testing data alignment always used
  `lambda = 0`, i.e. no penalty).
- `prep_testing_data` now returns a `call` component recording the `lambda`,
  `penalty_method`, and `optim_method` used for alignment.
- `prep_training_data` now validates `lambda` and `penalty_method` before
  passing them to `fdasrvf::time_warping`, and the documentation of `lambda`
  describes its role as the weight on the alignment penalty.

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
