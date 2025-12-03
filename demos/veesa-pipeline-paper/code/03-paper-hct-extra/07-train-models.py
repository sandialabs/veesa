#!/usr/bin/env python

# Contains: Train models using ESA fPCA results
# Working Directory: The location of this script (code/hct-extra)

# Load packages
import pickle
import time
import veesa_functions as veesa

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

# Train model using jfPCA/hfPCA/vfPCA results and compute performance on training data
for s in sparam:
    
  # Print sparam to keep track of progress
  print("Working on sparam:" + str(s))
  
  # Load fPCA results
  train_jfpca = veesa.load_object(folder_path=fp_res, train_test="train", stage="jfpca", sparam=s)
  train_vfpca = veesa.load_object(folder_path=fp_res, train_test="train", stage="vfpca", sparam=s)
  train_hfpca = veesa.load_object(folder_path=fp_res, train_test="train", stage="hfpca", sparam=s)
  
  # Prepare the fpcs for training
  train_x_jfpca = train_jfpca.coef
  train_x_vfpca = train_vfpca.coef
  train_x_hfpca = train_hfpca.coef

  ### jfPCA ----------------------------------------------------------------
  
  # Start clock
  start_time = time.time()

  # Train neural network, compute performance metrics, and save results
  veesa.apply_model(
    x=train_x_jfpca, 
    y=train_y, 
    analysis_name="jfpca", 
    sparam=s,
    folder_path=fp_res,
    seed=20211213
  )
  
  # End clock
  end_time = time.time()
  
  # Compute duration and save
  execution_time = end_time - start_time
  veesa.save_object(
    obj=execution_time,
    folder_path=fp_res,
    train_test="train",
    stage="time-model-jfpca",
    sparam=s
  )
  
  ### vfPCA ----------------------------------------------------------------
  
  # Start clock
  start_time = time.time()
  
  # Train neural network, compute performance metrics, and save results
  veesa.apply_model(
    x=train_x_vfpca, 
    y=train_y, 
    analysis_name="vfpca", 
    sparam=s, 
    folder_path=fp_res,
    seed=20211213
  )
  
  # End clock
  end_time = time.time()
  
  # Compute duration and save
  execution_time = end_time - start_time
  veesa.save_object(
    obj=execution_time,
    folder_path=fp_res,
    train_test="train",
    stage="time-model-vfpca",
    sparam=s
  )
  
  ### hfPCA ----------------------------------------------------------------
  
  # Start clock
  start_time = time.time()
  
  # Train neural network, compute performance metrics, and save results
  start_time = time.time()
  veesa.apply_model(
    x=train_x_hfpca, 
    y=train_y, 
    analysis_name="hfpca", 
    sparam=s, 
    folder_path=fp_res,
    seed=20211213
  )
  
  # End clock
  end_time = time.time()
  
  # Compute duration and save
  execution_time = end_time - start_time
  veesa.save_object(
    obj=execution_time,
    folder_path=fp_res,
    train_test="train",
    stage="time-model-hfpca",
    sparam=s
  )
