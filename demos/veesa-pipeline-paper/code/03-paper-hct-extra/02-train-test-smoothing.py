#!/usr/bin/env python

# Contains: Smoothing of HCT Data
# Working Directory: The location of this script (code/hct-extra)

# Load packages
import fdasrsf as fs
import numpy as np
import pickle
import time
import veesa_functions as veesa

# Specify file paths
fp_data = "../../data/"
fp_res = "../../results/hct/"

# Load H-CT data
train = pickle.load(open(fp_data + "hct-train.pkl", "rb"))
test = pickle.load(open(fp_data + "hct-test.pkl", "rb"))

# Prepare feature matrix for smoothing
train_functions = np.array(
  train[["id","frequency","value"]].pivot(index = "id", columns = "frequency")
).T
test_functions = np.array(
  test[["id","frequency","value"]].pivot(index = "id", columns = "frequency")
).T

# Specify smoothing parameters
sparam = [1, 5, 10, 15, 20, 25]

# Apply smoothing and save results
for s in sparam:
  
  print("Working on sparam:" + str(s))
    
  # Training data -------------------------------------------------------------

  # Start clock
  start_time = time.time()

  # Smooth data
  train_smoothed = fs.smooth_data(f=train_functions, sparam=s)

  # End clock
  end_time = time.time()

  # Compute duration and save
  veesa.save_object(
    obj=train_smoothed,
    folder_path=fp_res,
    train_test="train",
    stage="smoothed",
    sparam=s
  )

  # Compute duration and save
  execution_time = end_time - start_time
  veesa.save_object(
    obj=execution_time,
    folder_path=fp_res,
    train_test="train",
    stage="time-smoothed",
    sparam=s
  )

  # Testing data  -------------------------------------------------------------

  # Start clock
  start_time = time.time()

  # Smooth data
  test_smoothed = fs.smooth_data(f=test_functions, sparam=s)

  # End clock
  end_time = time.time()

  # Save smoothed data
  veesa.save_object(
    obj=test_smoothed,
    folder_path=fp_res,
    train_test="test",
    stage="smoothed",
    sparam=s
  )

  # Compute duration and save
  execution_time = end_time - start_time
  veesa.save_object(
    obj=execution_time,
    folder_path=fp_res,
    train_test="test",
    stage="time-smoothed",
    sparam=s
  )

