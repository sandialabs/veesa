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

## Subset Aligned Functions ---------------------------------------------------

# Specify smoothing parameters
sparam = [1, 5, 10, 15, 20, 25] 

# Subset for each sparam
for s in sparam:
  
  # Print sparam to keep track of progress
  print("Working on sparam: " + str(s))
  
  # Load aligned and warping functions
  train_aligned_warping = veesa.load_object(folder_path=fp_res, train_test="train", stage="aligned", sparam=s)
  train_aligned_warping_centered = veesa.load_object(folder_path=fp_res, train_test="train", stage="aligned-centered", sparam=s)
  
  # Convert functions to data frame
  train_aligned_df = pd.DataFrame(train_aligned_warping.fn.transpose())
  train_warping_df = pd.DataFrame(train_aligned_warping.gam.transpose())
  train_aligned_centered_df = pd.DataFrame(train_aligned_warping_centered.fn.transpose())
  train_warping_centered_df = pd.DataFrame(train_aligned_warping_centered.gam.transpose())
  
  # Add IDs
  train_aligned_df["id"] = train_ids
  train_warping_df["id"] = train_ids
  train_aligned_centered_df["id"] = train_ids
  train_warping_centered_df["id"] = train_ids
  
  # Subset the functions
  train_aligned_sub = train_aligned_df[train_aligned_df["id"].isin(train_sub_ids)]
  train_warping_sub = train_warping_df[train_warping_df["id"].isin(train_sub_ids)]
  train_aligned_centered_sub = train_aligned_centered_df[train_aligned_centered_df["id"].isin(train_sub_ids)]
  train_warping_centered_sub = train_warping_centered_df[train_warping_centered_df["id"].isin(train_sub_ids)]
  
  # Save the subsets
  veesa.save_object(obj=train_aligned_sub, folder_path=fp_res, train_test="train", stage="aligned", sparam=s, sub=True)
  veesa.save_object(obj=train_warping_sub, folder_path=fp_res, train_test="train", stage="warping", sparam=s, sub=True)
  veesa.save_object(obj=train_aligned_centered_sub, folder_path=fp_res, train_test="train", stage="aligned-centered", sparam=s, sub=True)
  veesa.save_object(obj=train_warping_centered_sub, folder_path=fp_res, train_test="train", stage="warping-centered", sparam=s, sub=True)
  
