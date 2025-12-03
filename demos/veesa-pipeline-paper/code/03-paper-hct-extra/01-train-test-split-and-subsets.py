#!/usr/bin/env python

# Contains: Preparation of training and testing H-CT data
# Working Directory: The location of this script (code/hct-extra)

# Load packages
import numpy as np
import pandas as pd
import pickle
import time

# Specify file paths
fp_data = "../../data/"
fp_res = "../../results/hct/"

# Load prepared H-CT data
#hct_data = pickle.load(open(fp_data + "hct-clean.pkl", "rb"))
hct_data = pickle.load(open(fp_data + "hct-clean-example.pkl", "rb"))

## Create Training/Testing Splits ---------------------------------------------

# Record the start time
start_time = time.time()

# Extract the training/testing data
train = hct_data[hct_data["dataset"] == "train"]
test = hct_data[hct_data["dataset"] == "test"]

# Compute duration and save
end_time = time.time()
execution_time = end_time - start_time
pickle.dump(execution_time , open(fp_res + "hct-both-time-split.pkl", "wb"))

# Save training/testing data
pickle.dump(train, open(fp_data + "hct-train.pkl", "wb"))
pickle.dump(test, open(fp_data + "hct-test.pkl", "wb"))

## Compute Data Dimensions ----------------------------------------------------
 
# Subset the training data by material
train_h2o = train[train["material"] == "h2o"]
train_exp = train[train["material"] == "explosive"]
train_hp100 = train[train["material"] == "hp100"]
train_hp50 = train[train["material"] == "hp50"]
train_hp10 = train[train["material"] == "hp10"]

# Put the traing dimensions in a dictionary
train_dims = {
  "train": train["id"].unique().shape[0],
  "h2o": train_h2o["id"].unique().shape[0],
  "exp": train_exp["id"].unique().shape[0],
  "hp100": train_hp100["id"].unique().shape[0],
  "hp50": train_hp50["id"].unique().shape[0],
  "hp10": train_hp10["id"].unique().shape[0]
}

# Subset the testing data by material
test_h2o = test[test["material"] == "h2o"]
test_exp = test[test["material"] == "explosive"]
test_hp100 = test[test["material"] == "hp100"]
test_hp50 = test[test["material"] == "hp50"]
test_hp10 = test[test["material"] == "hp10"]

# Put the testing dimensions in a dictionary
test_dims = {
  "test": test["id"].unique().shape[0],
  "h2o": test_h2o["id"].unique().shape[0],
  "exp": test_exp["id"].unique().shape[0],
  "hp100": test_hp100["id"].unique().shape[0],
  "hp50": test_hp50["id"].unique().shape[0],
  "hp10": test_hp10["id"].unique().shape[0]
}

# Save the dimensions
pickle.dump(train_dims, open(fp_data + "hct-train-dims.pkl", "wb"))
pickle.dump(test_dims, open(fp_data + "hct-test-dims.pkl", "wb"))

## Prepare Training/Testing Subsets -------------------------------------------

# Code commented since not needed for example data (but was used in analysis
# for the paper)

# Function for extracting a subset of the observations in the
# training data with a specific material
# def get_subset_df(df_full, material, n_samples, seed):
# 
#     # Select material observations
#     df_material = df_full[df_full["material"] == material]
# 
#     # Select a subset of IDs
#     np.random.seed(seed)
#     ids_sub = np.random.choice(
#       df_material["id"].unique(),
#       n_samples,
#       replace=False
#     )
# 
#     # Select the observations with the selected IDs
#     df_sub = df_material[df_material["id"].isin(ids_sub)]
# 
#     # Return the data
#     return df_sub

# Specify number of observations per subset
# nobs_sub = 1000

# Get subsets for each of the five materials (Training)
# train_h2o_sub = get_subset_df(train, "h2o", nobs_sub, 2021)
# train_exp_sub = get_subset_df(train, "explosive", nobs_sub, 2021)
# train_hp100_sub = get_subset_df(train, "hp100", nobs_sub, 2021)
# train_hp50_sub = get_subset_df(train, "hp50", nobs_sub, 2021)
# train_hp10_sub = get_subset_df(train, "hp10", nobs_sub, 2021)

# # Join the material subsets
# train_sub = pd.concat([
#     train_h2o_sub,
#     train_exp_sub,
#     train_hp100_sub,
#     train_hp50_sub,
#     train_hp10_sub
# ])

# # Get subsets for each of the five materials (testing)
# test_h2o_sub = get_subset_df(test, "h2o", nobs_sub, 2021)
# test_exp_sub = get_subset_df(test, "explosive", nobs_sub, 2021)
# test_hp100_sub = get_subset_df(test, "hp100", nobs_sub, 2021)
# test_hp50_sub = get_subset_df(test, "hp50", nobs_sub, 2021)
# test_hp10_sub = get_subset_df(test, "hp10", nobs_sub, 2021)

# Join the material subsets
# test_sub = pd.concat([
#     test_h2o_sub,
#     test_exp_sub,
#     test_hp100_sub,
#     test_hp50_sub,
#     test_hp10_sub
# ])

# For example data only
train_sub = train.copy()
test_sub = test.copy()

# Save the data
pickle.dump(train_sub, open(fp_data + "hct-sub-train.pkl", "wb"))
pickle.dump(test_sub, open(fp_data + "hct-sub-test.pkl", "wb"))
