#!/usr/bin/env python

# Application of PFI to models using ESA fPCA results (with test data)
# Author: Katherine Goode
# Date created: February 7, 2022

# Load packages
import fdasrsf as fs
import hct_functions as hct
import numpy as np
import pickle
from sklearn.inspection import permutation_importance

# Load H-CT data
test = pickle.load(open("../data/hct-test.pkl", "rb"))

# Prepare the response variables for training
test_y = test[['id','material']].drop_duplicates()['material']

# Create a vector with the normalized frequencies
freq_norm = test.frequency_norm.unique()

# Specify smoothing parameters
sparam = [1, 5, 10, 15, 20, 25]

# Compute PFI
for s in sparam:
    
  # Print sparam to keep track of progress
  print("Working on sparam:" + str(s))
  
  # Load jfPCA results
  test_jfpca = hct.load_object(data_results="data", train_test="test", stage="aligned-jfpca", sparam=s)
  test_vfpca = hct.load_object(data_results="data", train_test="test", stage="aligned-vfpca", sparam=s)
  test_hfpca = hct.load_object(data_results="data", train_test="test", stage="aligned-hfpca", sparam=s)

  # Load the model
  nn_jfpca = hct.load_object(data_results="results", train_test="train", stage="nn-jfpca", sparam=s)
  nn_vfpca = hct.load_object(data_results="results", train_test="train", stage="nn-vfpca", sparam=s)
  nn_hfpca = hct.load_object(data_results="results", train_test="train", stage="nn-hfpca", sparam=s)

  # Compute and PFI for the neural network using accuracy (jfPCA)
  pfi_jfpca = permutation_importance(
    estimator=nn_jfpca, 
    X=test_jfpca['coef'], 
    y=test_y,
    scoring='accuracy', 
    n_repeats=5, 
    n_jobs=5, 
    random_state=20211213
  )
  hct.save_object(obj=pfi_jfpca, data_results="results", train_test="test", stage="pfi-jfpca", sparam=s)
  
  # Compute and PFI for the neural network using accuracy (vfPCA)
  pfi_vfpca = permutation_importance(
    estimator=nn_vfpca, 
    X=test_vfpca['coef'], 
    y=test_y,
    scoring='accuracy', 
    n_repeats=5, 
    n_jobs=5, 
    random_state=20211213
  )
  hct.save_object(obj=pfi_vfpca, data_results="results", train_test="test", stage="pfi-vfpca", sparam=s)
  
  # Compute and PFI for the neural network using accuracy (hfPCA)
  pfi_hfpca = permutation_importance(
    estimator=nn_hfpca, 
    X=test_hfpca['coef'], 
    y=test_y,
    scoring='accuracy', 
    n_repeats=5, 
    n_jobs=5, 
    random_state=20211213
  )
  hct.save_object(obj=pfi_hfpca, data_results="results", train_test="test", stage="pfi-hfpca", sparam=s)
