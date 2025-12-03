#!/usr/bin/env python

# Contains: Compute performance metrics on test data for the ESA fPCA models
# Working Directory: The location of this script (code/hct-extra)

# Load packages
import pickle
import time
import veesa_functions as veesa

# Specify file paths
fp_data = "../../data/"
fp_res = "../../results/hct/"

# Load H-CT data
test = pickle.load(open(fp_data + "hct-test.pkl", "rb"))

# Prepare the response variables for training
test_y = test[["id","material"]].drop_duplicates()["material"]

# Create a vector with the normalized frequencies
freq_norm = test.frequency_norm.unique()

# Specify smoothing parameters
sparam = [1, 5, 10, 15, 20, 25]

# Apply alignment and fPCA to test data based on training data
for s in sparam:

  # Print sparam to keep track of progress
  print("Working on sparam:" + str(s))
  
  ### Load data  -----------------------------------------------------------
  
  # Load smoothed test data
  test_smoothed = veesa.load_object(
    folder_path=fp_res, 
    train_test="test", 
    stage="smoothed", 
    sparam=s
  )
  
  # Load aligned data
  train_aligned = veesa.load_object(
    folder_path=fp_res, 
    train_test="train",
    stage="aligned", 
    sparam=s
  )
  
  ### jfPCA ----------------------------------------------------------------
  
  # Load fPCA results
  train_jfpca = veesa.load_object(
    folder_path=fp_res, 
    train_test="train",
    stage="jfpca",
    sparam=s
  )
  
  # Start clock
  start_time = time.time()
  
  # Prepare test data
  test_aligned_jfpca = veesa.prep_testing_data(
    f=test_smoothed, 
    time=freq_norm, 
    aligned_train=train_aligned, 
    fpca_train=train_jfpca, 
    fpca_method="jfpca", 
    omethod="DP"
  )
  
  # End clock
  end_time = time.time()
  
  # Compute duration and save
  execution_time = end_time - start_time
  veesa.save_object(
    obj=execution_time,
    folder_path=fp_res,
    train_test="test",
    stage="time-aligned-jfpca",
    sparam=s
  )
  
  # Save aligned test data
  veesa.save_object(
    obj=test_aligned_jfpca,
    folder_path=fp_res, 
    train_test="test", 
    stage="aligned-jfpca",
    sparam=s
  )
  
  ### vfPCA ----------------
  
  # Load fPCA results
  train_vfpca = veesa.load_object(
    folder_path=fp_res, 
    train_test="train", 
    stage="vfpca", 
    sparam=s
  )
  
  # Start clock
  start_time = time.time()
  
  # Prepare test data
  test_aligned_vfpca = veesa.prep_testing_data(
    f=test_smoothed, 
    time=freq_norm, 
    aligned_train=train_aligned, 
    fpca_train=train_vfpca, 
    fpca_method="vfpca", 
    omethod="DP"
  )
  
  # End clock
  end_time = time.time()
  
  # Compute duration and save
  execution_time = end_time - start_time
  veesa.save_object(
    obj=execution_time,
    folder_path=fp_res,
    train_test="test",
    stage="time-aligned-vfpca",
    sparam=s
  )
  
  # Save aligned test data
  veesa.save_object(
    obj=test_aligned_vfpca,
    folder_path=fp_res, 
    train_test="test", 
    stage="aligned-vfpca",
    sparam=s
  )
  
  ### hfPCA ----------------
  
  # Load fPCA results
  train_hfpca = veesa.load_object(
    folder_path=fp_res, 
    train_test="train",
    stage="hfpca",
    sparam=s
  )
  
  # Start clock
  start_time = time.time()
  
  # Prepare test data
  test_aligned_hfpca = veesa.prep_testing_data(
    f=test_smoothed, 
    time=freq_norm, 
    aligned_train=train_aligned, 
    fpca_train=train_hfpca, 
    fpca_method="hfpca", 
    omethod="DP"
  )
  
  # End clock
  end_time = time.time()
  
  # Compute duration and save
  execution_time = end_time - start_time
  veesa.save_object(
    obj=execution_time,
    folder_path=fp_res,
    train_test="test",
    stage="time-aligned-hfpca",
    sparam=s
  )
  
  # Save aligned test data
  veesa.save_object(
    obj=test_aligned_hfpca,
    folder_path=fp_res,
    train_test="test", 
    stage="aligned-hfpca",
    sparam=s
  )
