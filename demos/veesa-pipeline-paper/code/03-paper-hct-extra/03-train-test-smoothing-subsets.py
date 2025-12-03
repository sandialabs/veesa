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

## Subset Smoothed Functions --------------------------------------------------

# Specify smoothing parameters
sparam = [1, 5, 10, 15, 20, 25] 

# Subset for each sparam
for s in sparam:
  
  # Print sparam to keep track of progress
  print("Working on sparam: " + str(s))
  
  # Load smoothed data
  train_smoothed = veesa.load_object(folder_path=fp_res, train_test="train", stage="smoothed", sparam=s)
  test_smoothed = veesa.load_object(folder_path=fp_res, train_test="test", stage="smoothed", sparam=s)
  
  # Convert functions to data frame
  train_smoothed_df = pd.DataFrame(train_smoothed.transpose())
  test_smoothed_df = pd.DataFrame(test_smoothed.transpose())
  
  # Add IDs
  train_smoothed_df["id"] = train_ids
  test_smoothed_df["id"] = test_ids
  
  # Subset the functions
  train_smoothed_sub = train_smoothed_df[train_smoothed_df["id"].isin(train_sub_ids)]
  test_smoothed_sub = test_smoothed_df[test_smoothed_df["id"].isin(test_sub_ids)]

  # Save the subsets
  veesa.save_object(obj=train_smoothed_sub, folder_path=fp_res, train_test="train", stage="smoothed", sparam=s, sub=True)
  veesa.save_object(obj=test_smoothed_sub, folder_path=fp_res, train_test="test", stage="smoothed", sparam=s, sub=True)

