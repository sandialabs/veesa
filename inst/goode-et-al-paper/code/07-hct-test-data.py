#!/usr/bin/env python

# Compute performance metrics on test data for the ESA fPCA models
# Author: Katherine Goode
# Date created: December 13, 2021

# Load packages
import fdasrsf as fs
import hct_functions as hct
import numpy as np
import pickle
from sklearn.metrics import accuracy_score

# Load H-CT data
test = pickle.load(open("../data/hct-test.pkl", "rb"))

# Prepare the response variables for training
test_y = test[['id','material']].drop_duplicates()['material']

# Create a vector with the normalized frequencies
freq_norm = test.frequency_norm.unique()

# Specify smoothing parameters
sparam = [1, 5, 10, 15, 20, 25]

# Apply alignment and fPCA to test data based on training data
for s in sparam:

  # Print sparam to keep track of progress
  print("Working on sparam:" + str(s))
  
  # Load aligned data
  train_aligned = hct.load_object(data_results = "data", train_test="train", stage="aligned", sparam=s)
  
  # Load fPCA results
  train_jfpca = hct.load_object(data_results = "data", train_test="train", stage="jfpca", sparam=s)
  train_vfpca = hct.load_object(data_results = "data", train_test="train", stage="vfpca", sparam=s)
  train_hfpca = hct.load_object(data_results = "data", train_test="train", stage="hfpca", sparam=s)
  
  # Load smoothed test data
  test_smoothed = hct.load_object(data_results = "data", train_test="test", stage="smoothed", sparam=s)

  # Prepare test data (and save results)
  test_aligned_jfpca = hct.prep_testing_data(f=test_smoothed, time=freq_norm, aligned_train=train_aligned, fpca_train=train_jfpca, fpca_method="jfpca", omethod="DP")
  hct.save_object(
    obj=test_aligned_jfpca,
    data_results="data", 
    train_test="test", 
    stage="aligned-jfpca",
    sparam=s
  )
  test_aligned_vfpca = hct.prep_testing_data(f=test_smoothed, time=freq_norm, aligned_train=train_aligned, fpca_train=train_vfpca, fpca_method="vfpca", omethod="DP")
  hct.save_object(
    obj=test_aligned_vfpca,
    data_results="data", 
    train_test="test", 
    stage="aligned-vfpca",
    sparam=s
  )
  test_aligned_hfpca = hct.prep_testing_data(f=test_smoothed, time=freq_norm, aligned_train=train_aligned, fpca_train=train_hfpca, fpca_method="hfpca", omethod="DP")
  hct.save_object(
    obj=test_aligned_hfpca,
    data_results="data", 
    train_test="test", 
    stage="aligned-hfpca",
    sparam=s
  )

  # Load the models
  nn_jfpca = hct.load_object(data_results="results", train_test="train", stage="nn-jfpca", sparam=s)
  nn_vfpca = hct.load_object(data_results="results", train_test="train", stage="nn-vfpca", sparam=s)
  nn_hfpca = hct.load_object(data_results="results", train_test="train", stage="nn-hfpca", sparam=s)
  
  # Get predictions and accuracy on test data
  test_pred_jfpca = nn_jfpca.predict(X = test_aligned_jfpca['coef'])
  test_acc_jfpca = accuracy_score(y_true = test_y, y_pred = test_pred_jfpca)
  test_pred_vfpca = nn_vfpca.predict(X = test_aligned_vfpca['coef'])
  test_acc_vfpca = accuracy_score(y_true = test_y, y_pred = test_pred_vfpca)
  test_pred_hfpca = nn_hfpca.predict(X = test_aligned_hfpca['coef'])
  test_acc_hfpca = accuracy_score(y_true = test_y, y_pred = test_pred_hfpca)
  
  # Join the predictions and metrics in a dictionary (and save results)
  test_res_jfpca = {"preds": test_pred_jfpca, "acc": test_acc_jfpca}
  hct.save_object(
    obj=test_res_jfpca,
    data_results="results", 
    train_test="test", 
    stage="pred-and-metrics-jfpca", 
    sparam=s
  )
  test_res_vfpca = {"preds": test_pred_vfpca, "acc": test_acc_vfpca}
  hct.save_object(
    obj=test_res_vfpca,
    data_results="results", 
    train_test="test", 
    stage="pred-and-metrics-vfpca", 
    sparam=s
  )
  test_res_hfpca = {"preds": test_pred_hfpca, "acc": test_acc_hfpca}
  hct.save_object(
    obj=test_res_hfpca,
    data_results="results", 
    train_test="test", 
    stage="pred-and-metrics-hfpca", 
    sparam=s
  )
