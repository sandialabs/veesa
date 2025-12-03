#!/usr/bin/env python

# Contains: Extract subsets from aligned data
# Working Directory: The location of this script (code/hct-extra)

# Load packages
import pandas as pd
import pickle
import veesa_functions as veesa

# Specify file paths
fp_data = "../../data/"
fp_res = "../../results/hct/"

## Load Datasets --------------------------------------------------------------

# Load H-CT data
train = pickle.load(open(fp_data + "hct-train.pkl", "rb"))
test = pickle.load(open(fp_data + "hct-test.pkl", "rb"))

# Load subset data
train_sub = pickle.load(open(fp_data + "hct-train.pkl", "rb"))
test_sub = pickle.load(open(fp_data + "hct-test.pkl", "rb"))

# Extract all IDs
train_ids = train["id"].unique()
test_ids = test["id"].unique()

# Extract IDs from subsets
train_sub_ids = train_sub["id"].unique()
test_sub_ids = test_sub["id"].unique()

## Subset Aligned/Warping Functions -------------------------------------------

# Specify smoothing parameters
sparam = [1, 5, 10, 15, 20, 25] 

# Subset for each sparam
for s in sparam:
  
  # Print sparam to keep track of progress
  print("Working on sparam: " + str(s))
  
  # Load aligned and warping functions
  test_aligned_warping = veesa.load_object(folder_path=fp_res, train_test="test", stage="aligned-jfpca", sparam=s)
  
  # Convert functions to data frame
  test_aligned_df = pd.DataFrame(test_aligned_warping["fn"].transpose())
  test_warping_df = pd.DataFrame(test_aligned_warping["gam"].transpose())
  
  # Add IDs
  test_aligned_df["id"] = test_ids
  test_warping_df["id"] = test_ids

  # Subset the functions
  test_aligned_sub = test_aligned_df[test_aligned_df["id"].isin(test_sub_ids)]
  test_warping_sub = test_warping_df[test_warping_df["id"].isin(test_sub_ids)]

  # Save the subsets
  veesa.save_object(obj=test_aligned_sub, folder_path=fp_res, train_test="test", stage="aligned", sparam=s, sub=True)
  veesa.save_object(obj=test_warping_sub, folder_path=fp_res, train_test="test", stage="warping", sparam=s, sub=True)
