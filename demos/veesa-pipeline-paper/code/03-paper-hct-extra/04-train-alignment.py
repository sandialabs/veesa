#!/usr/bin/env python

# Contains: Alignment of smoothed H-CT training data
# Working Directory: The location of this script (code/hct-extra)

# Load packages
import fdasrsf as fs
import pickle
import time
import veesa_functions as veesa

# Specify file paths
fp_data = "../../data/"
fp_res = "../../results/hct/"

# Load H-CT training data
train = pickle.load(open(fp_data + "hct-train.pkl", "rb"))

# Create a vector with the normalized frequencies
freq_norm = train.frequency_norm.unique()

# Specify smoothing parameters
sparam = [1, 5, 10, 15, 20, 25]

# Align training functions
for s in sparam:
  
  # Print sparam to keep track of progress
  print("Working on sparam:" + str(s))
    
  # Load the smoothed train/test data
  train_smoothed = veesa.load_object(
    folder_path=fp_res,
    train_test="train", 
    stage="smoothed",
    sparam=s
  )

  # Start clock
  start_time = time.time()

  # Align data
  train_aligned = fs.time_warping.fdawarp(train_smoothed, freq_norm)
  train_aligned.srsf_align(parallel=True, cores=-1, center=False, omethod="DP")
  
  # End clock
  end_time = time.time()
  
  # Compute duration and save
  execution_time = end_time - start_time
  veesa.save_object(
    obj=execution_time,
    folder_path=fp_res,
    train_test="train",
    stage="time-aligned",
    sparam=s
  )
  
  # Save results
  veesa.save_object(
    obj=train_aligned, 
    folder_path=fp_res, 
    train_test="train", 
    stage="aligned", 
    sparam=s
  )
  
  # Apply centering to aligned data (and save results)
  train_aligned_centered = veesa.center_warping_funs(train_aligned)
  veesa.save_object(
    obj=train_aligned_centered,
    folder_path=fp_res,
    train_test="train", 
    stage="aligned-centered", 
    sparam=s
  )
