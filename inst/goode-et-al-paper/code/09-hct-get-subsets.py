#!/usr/bin/env python

# Extract subsets from datasets of interest
# Author: Katherine Goode
# Date created: December 6, 2021

# # Load packages
import numpy as np
import hct_functions as hct
import pandas as pd
import pickle
 
## Load Datasets ----------------------------------------------------------------------------------

# Load the clean version of the HCT data
train = pickle.load(open("../data/hct-train.pkl", "rb"))
test = pickle.load(open("../data/hct-test.pkl", "rb"))

## Compute Data Dimensions ------------------------------------------------------------------------

# Subset the training data by material
train_h2o = train[train['material'] == 'h2o']
train_exp = train[train['material'] == 'explosive']
train_hp100 = train[train['material'] == 'hp100']
train_hp50 = train[train['material'] == 'hp50']
train_hp10 = train[train['material'] == 'hp10']

# Put the traing dimensions in a dictionary
train_dims = {
  "train": train['id'].unique().shape[0],
  "h2o": train_h2o['id'].unique().shape[0],
  "exp": train_exp['id'].unique().shape[0],
  "hp100": train_hp100['id'].unique().shape[0],
  "hp50": train_hp50['id'].unique().shape[0],
  "hp10": train_hp10['id'].unique().shape[0]
}

# Subset the testing data by material
test_h2o = test[test['material'] == 'h2o']
test_exp = test[test['material'] == 'explosive']
test_hp100 = test[test['material'] == 'hp100']
test_hp50 = test[test['material'] == 'hp50']
test_hp10 = test[test['material'] == 'hp10']

# Put the testing dimensions in a dictionary
test_dims = {
  "test": test['id'].unique().shape[0],
  "h2o": test_h2o['id'].unique().shape[0],
  "exp": test_exp['id'].unique().shape[0],
  "hp100": test_hp100['id'].unique().shape[0],
  "hp50": test_hp50['id'].unique().shape[0],
  "hp10": test_hp10['id'].unique().shape[0]
}

# Save the dimensions
pickle.dump(train_dims, open('../data/hct-train-dims.pkl', 'wb'))
pickle.dump(test_dims, open('../data/hct-test-dims.pkl', 'wb'))

## Prepare Training/Testing Subsets ---------------------------------------------------------------

# Function for extracting a subset of the observations in the
# training data with a specific material
def get_subset_df(df_full, material, n_samples, seed):

    # Select material observations
    df_material = df_full[df_full['material'] == material]

    # Select a subset of IDs
    np.random.seed(seed)
    ids_sub = np.random.choice(df_material['id'].unique(), n_samples, replace=False)

    # Select the observations with the selected IDs
    df_sub = df_material[df_material['id'].isin(ids_sub)]

    # Return the data
    return df_sub

# Specify number of observations per subset
nobs_sub = 1000

# Get subsets for each of the five materials (Training)
train_h2o_sub = get_subset_df(train, "h2o", nobs_sub, 2021)
train_exp_sub = get_subset_df(train, "explosive", nobs_sub, 2021)
train_hp100_sub = get_subset_df(train, "hp100", nobs_sub, 2021)
train_hp50_sub = get_subset_df(train, "hp50", nobs_sub, 2021)
train_hp10_sub = get_subset_df(train, "hp10", nobs_sub, 2021)

# Join the material subsets
train_sub = pd.concat([
    train_h2o_sub,
    train_exp_sub,
    train_hp100_sub,
    train_hp50_sub,
    train_hp10_sub
])

# Get subsets for each of the five materials (testing)
test_h2o_sub = get_subset_df(test, "h2o", nobs_sub, 2021)
test_exp_sub = get_subset_df(test, "explosive", nobs_sub, 2021)
test_hp100_sub = get_subset_df(test, "hp100", nobs_sub, 2021)
test_hp50_sub = get_subset_df(test, "hp50", nobs_sub, 2021)
test_hp10_sub = get_subset_df(test, "hp10", nobs_sub, 2021)

# Join the material subsets
test_sub = pd.concat([
    test_h2o_sub,
    test_exp_sub,
    test_hp100_sub,
    test_hp50_sub,
    test_hp10_sub
])

# Save the data
pickle.dump(train_sub, open('../data/hct-sub-train.pkl', 'wb'))
pickle.dump(test_sub, open('../data/hct-sub-test.pkl', 'wb'))

## Subset Smoothed/Aligned/Warping Functions ------------------------------------------------------

# Extract all IDs
train_ids = train['id'].unique()
test_ids = test['id'].unique()

# Extract IDs from subsets
train_sub_ids = train_sub['id'].unique()
test_sub_ids = test_sub['id'].unique()

# Specify smoothing parameters
sparam = [1, 5, 10, 15, 20, 25]

# Subset for each sparam
for s in sparam:
  
  # Print sparam to keep track of progress
  print("Working on sparam: " + str(s))
  
  # Load smoothed data
  train_smoothed = hct.load_object(data_results="data", train_test="train", stage="smoothed", sparam=s)
  test_smoothed = hct.load_object(data_results="data", train_test="test", stage="smoothed", sparam=s)
  
  # Load aligned and warping functions
  train_aligned_warping = hct.load_object(data_results="data", train_test="train", stage="aligned", sparam=s)
  train_aligned_warping_centered = hct.load_object(data_results="data", train_test="train", stage="aligned-centered", sparam=s)
  test_aligned_warping = hct.load_object(data_results="data", train_test="test", stage="aligned-jfpca", sparam=s)
  
  # Convert functions to data frame
  train_smoothed_df = pd.DataFrame(train_smoothed.transpose())
  train_aligned_df = pd.DataFrame(train_aligned_warping.fn.transpose())
  train_warping_df = pd.DataFrame(train_aligned_warping.gam.transpose())
  train_aligned_centered_df = pd.DataFrame(train_aligned_warping_centered.fn.transpose())
  train_warping_centered_df = pd.DataFrame(train_aligned_warping_centered.gam.transpose())
  test_smoothed_df = pd.DataFrame(test_smoothed.transpose())
  test_aligned_df = pd.DataFrame(test_aligned_warping['fn'].transpose())
  test_warping_df = pd.DataFrame(test_aligned_warping['gam'].transpose())
  
  # Add IDs
  train_smoothed_df['id'] = train_ids
  train_aligned_df['id'] = train_ids
  train_warping_df['id'] = train_ids
  train_aligned_centered_df['id'] = train_ids
  train_warping_centered_df['id'] = train_ids
  test_smoothed_df['id'] = test_ids
  test_aligned_df['id'] = test_ids
  test_warping_df['id'] = test_ids

  # Subset the functions
  train_smoothed_sub = train_smoothed_df[train_smoothed_df['id'].isin(train_sub_ids)]
  train_aligned_sub = train_aligned_df[train_aligned_df['id'].isin(train_sub_ids)]
  train_warping_sub = train_warping_df[train_warping_df['id'].isin(train_sub_ids)]
  train_aligned_centered_sub = train_aligned_centered_df[train_aligned_centered_df['id'].isin(train_sub_ids)]
  train_warping_centered_sub = train_warping_centered_df[train_warping_centered_df['id'].isin(train_sub_ids)]
  test_smoothed_sub = test_smoothed_df[test_smoothed_df['id'].isin(test_sub_ids)]
  test_aligned_sub = test_aligned_df[test_aligned_df['id'].isin(test_sub_ids)]
  test_warping_sub = test_warping_df[test_warping_df['id'].isin(test_sub_ids)]

  # Save the subsets
  hct.save_object(obj=train_smoothed_sub, data_results="data", train_test="train", stage="smoothed", sparam=s, sub=True)
  hct.save_object(obj=train_aligned_sub, data_results="data", train_test="train", stage="aligned", sparam=s, sub=True)
  hct.save_object(obj=train_warping_sub, data_results="data", train_test="train", stage="warping", sparam=s, sub=True)
  hct.save_object(obj=train_aligned_centered_sub, data_results="data", train_test="train", stage="aligned-centered", sparam=s, sub=True)
  hct.save_object(obj=train_warping_centered_sub, data_results="data", train_test="train", stage="warping-centered", sparam=s, sub=True)
  hct.save_object(obj=test_smoothed_sub, data_results="data", train_test="test", stage="smoothed", sparam=s, sub=True)
  hct.save_object(obj=test_aligned_sub, data_results="data", train_test="test", stage="aligned", sparam=s, sub=True)
  hct.save_object(obj=test_warping_sub, data_results="data", train_test="test", stage="warping", sparam=s, sub=True)
