#!/usr/bin/env python

# Smoothing of HCT Data
# Author: Katherine Goode
# Date created: December 6, 2021

# Load packages
import fdasrsf as fs
import hct_functions as hct
import numpy as np
import pickle

# Load H-CT data
train = pickle.load(open("../data/hct-train.pkl", "rb"))
test = pickle.load(open("../data/hct-test.pkl", "rb"))

# Prepare feature matrix for smoothing
train_functions = np.array(train[['id','frequency','value']].pivot(index = 'id', columns = 'frequency')).T
test_functions = np.array(test[['id','frequency','value']].pivot(index = 'id', columns = 'frequency')).T

# Specify smoothing parameters
sparam = [1, 5, 10, 15, 20, 25]

# Apply smoothing and save results
for s in sparam:
  print("Working on sparam:" + str(s))
  train_smoothed = fs.smooth_data(f=train_functions, sparam=s)
  hct.save_object(
    obj=train_smoothed,
    data_results="data",
    train_test="train",
    stage="smoothed",
    sparam=s
  )
  test_smoothed = fs.smooth_data(f=test_functions, sparam=s)
  hct.save_object(
    obj=test_smoothed,
    data_results="data",
    train_test="test",
    stage="smoothed",
    sparam=s
  )
