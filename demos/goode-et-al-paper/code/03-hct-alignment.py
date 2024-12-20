#!/usr/bin/env python

# Alignment of smoothed H-CT training data
# Author: Katherine Goode
# Date created: December 6, 2021

# Load packages
import fdasrsf as fs
import hct_functions as hct
import numpy as np
import pickle

# Load H-CT training data
train = pickle.load(open("../data/hct-train.pkl", "rb"))

# Create a vector with the normalized frequencies
freq_norm = train.frequency_norm.unique()

# Specify smoothing parameters
sparam = [1, 5, 10, 15, 20, 25]

# Align training functions
for s in sparam:
  
  # Print sparam to keep track of progress
  print("Working on sparam:" + str(s))
  
  # Load the smoothed train/test data
  train_smoothed = hct.load_object(data_results="data", train_test="train", stage="smoothed", sparam=s)

  # Align data (and save results)
  train_aligned = fs.time_warping.fdawarp(train_smoothed, freq_norm)
  train_aligned.srsf_align(parallel=True, cores=-1, center=False, omethod="DP")
  hct.save_object(obj=train_aligned, data_results="data", train_test="train", stage="aligned", sparam=s)

  # Load aligned data
  train_aligned = hct.load_object(data_results = "data", train_test="train", stage="aligned", sparam=s)
  
  # Apply centering to aligned data (and save results)
  train_aligned_centered = hct.center_warping_funs(train_aligned)
  hct.save_object(obj=train_aligned_centered, data_results="data", train_test="train", stage="aligned-centered", sparam=s)
