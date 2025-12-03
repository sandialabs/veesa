#!/usr/bin/env python

# Contains: Application of PFI to models using ESA fPCA results (with train data)
# Working Directory: The location of this script (code/hct-extra)

# Load packages
import pickle
import time
import veesa_functions as veesa
from sklearn.inspection import permutation_importance

# Specify file paths
fp_data = "../../data/"
fp_res = "../../results/hct/"

# Load H-CT data
train = pickle.load(open(fp_data + "hct-train.pkl", "rb"))

# Prepare the response variables for training
train_y = train[['id','material']].drop_duplicates()['material']

# Create a vector with the normalized frequencies
freq_norm = train.frequency_norm.unique()

# Specify smoothing parameters
sparam = [1, 5, 10, 15, 20, 25]

# Compute PFI
for s in sparam:
    
  # Print sparam to keep track of progress
  print("Working on sparam:" + str(s))
  
  # Load jfPCA results
  train_jfpca = veesa.load_object(folder_path=fp_res, train_test="train", stage="jfpca", sparam=s)
  train_vfpca = veesa.load_object(folder_path=fp_res, train_test="train", stage="vfpca", sparam=s)
  train_hfpca = veesa.load_object(folder_path=fp_res, train_test="train", stage="hfpca", sparam=s)

  # Load the models
  nn_jfpca = veesa.load_object(folder_path=fp_res, train_test="train", stage="nn-jfpca", sparam=s)
  nn_vfpca = veesa.load_object(folder_path=fp_res, train_test="train", stage="nn-vfpca", sparam=s)
  nn_hfpca = veesa.load_object(folder_path=fp_res, train_test="train", stage="nn-hfpca", sparam=s)

  ### jfPCA ----------------------------------------------------------------
  
  # Start clock
  start_time = time.time()

  # Compute and PFI for the neural network using accuracy (jfPCA)
  pfi_jfpca = permutation_importance(
    estimator=nn_jfpca, 
    X=train_jfpca.coef, 
    y=train_y,
    scoring='accuracy', 
    n_repeats=5, 
    n_jobs=5, 
    random_state=20211213
  )
  
  # End clock
  end_time = time.time()
  
  # Compute duration and save
  execution_time = end_time - start_time
  veesa.save_object(
    obj=execution_time,
    folder_path=fp_res,
    train_test="train",
    stage="time-pfi-jfpca",
    sparam=s
  )
  
  # Save results
  veesa.save_object(
    obj=pfi_jfpca, 
    folder_path=fp_res, 
    train_test="train",
    stage="pfi-jfpca",
    sparam=s
  )
  
  ### vfPCA ----------------------------------------------------------------
  
  # Start clock
  start_time = time.time()
  
  # Compute and PFI for the neural network using accuracy (vfPCA)
  pfi_vfpca = permutation_importance(
    estimator=nn_vfpca, 
    X=train_vfpca.coef, 
    y=train_y,
    scoring='accuracy', 
    n_repeats=5, 
    n_jobs=5, 
    random_state=20211213
  )
  
  # End clock
  end_time = time.time()
  
  # Compute duration and save
  execution_time = end_time - start_time
  veesa.save_object(
    obj=execution_time,
    folder_path=fp_res,
    train_test="train",
    stage="time-pfi-vfpca",
    sparam=s
  )
  
  # Save results
  veesa.save_object(
    obj=pfi_vfpca,
    folder_path=fp_res, 
    train_test="train", 
    stage="pfi-vfpca", 
    sparam=s
  )
  
  ### hfPCA ----------------------------------------------------------------
  
  # Start clock
  start_time = time.time()
  
  # Compute and PFI for the neural network using accuracy (hfPCA)
  pfi_hfpca = permutation_importance(
    estimator=nn_hfpca, 
    X=train_hfpca.coef, 
    y=train_y,
    scoring='accuracy', 
    n_repeats=5, 
    n_jobs=5, 
    random_state=20211213
  )
  
  # End clock
  end_time = time.time()
  
  # Compute duration and save
  execution_time = end_time - start_time
  veesa.save_object(
    obj=execution_time,
    folder_path=fp_res,
    train_test="train",
    stage="time-pfi-hfpca",
    sparam=s
  )
  
  # Save results
  veesa.save_object(
    obj=pfi_hfpca, 
    folder_path=fp_res, 
    train_test="train", 
    stage="pfi-hfpca", 
    sparam=s
  )
