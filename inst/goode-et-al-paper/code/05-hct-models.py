#!/usr/bin/env python

# Train models using ESA fPCA results
# Author: Katherine Goode
# Date created: December 13, 2021

# Load packages
import fdasrsf as fs
import hct_functions as hct
import numpy as np
import pickle

# Load H-CT data
train = pickle.load(open("../data/hct-train.pkl", "rb"))

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
  train_jfpca = hct.load_object(data_results="data", train_test="train", stage="jfpca", sparam=s)
  train_vfpca = hct.load_object(data_results="data", train_test="train", stage="vfpca", sparam=s)
  train_hfpca = hct.load_object(data_results="data", train_test="train", stage="hfpca", sparam=s)
  
  # Prepare the fpcs for training
  train_x_jfpca = train_jfpca.coef
  train_x_vfpca = train_vfpca.coef
  train_x_hfpca = train_hfpca.coef

  # Train neural network, compute performance metrics, and save results
  hct.apply_model(x=train_x_jfpca, y=train_y, analysis_name="jfpca", sparam=s, seed=20211213)
  hct.apply_model(x=train_x_vfpca, y=train_y, analysis_name="vfpca", sparam=s, seed=20211213)
  hct.apply_model(x=train_x_hfpca, y=train_y, analysis_name="hfpca", sparam=s, seed=20211213)
