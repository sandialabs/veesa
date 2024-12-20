#!/usr/bin/env python

# Cross-sectional approach (pre-smoothing) to modeling H-CT data
# Author: Katherine Goode
# Date created: December 6, 2021

# Load packages
import fdasrsf as fs
import hct_functions as hct
import numpy as np
import pickle
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

# Load training and testing data
train = pickle.load(open("../data/hct-train.pkl", "rb"))
test = pickle.load(open("../data/hct-test.pkl", "rb"))

# Prepare response variables
train_y = train[['id','material']].drop_duplicates()['material']
test_y = test[['id','material']].drop_duplicates()['material']

# Prepare features
train_x = np.array(train[['id','frequency','value']].pivot(index = 'id', columns = 'frequency'))
test_x = np.array(test[['id','frequency','value']].pivot(index = 'id', columns = 'frequency'))

# Train neural network, compute performance metrics, and save results
hct.apply_model(x=train_x, y=train_y, analysis_name="cs-pre-smooth", sparam=0, seed=20211213)

# Load the model
nn = hct.load_object(data_results="results", train_test="train", stage="nn-cs-pre-smooth", sparam=0)

# Apply PFI
pfi = permutation_importance(
  estimator=nn, 
  X=train_x, 
  y=train_y,
  scoring='accuracy', 
  n_repeats=5, 
  n_jobs=5, 
  random_state=20211213
)

# Save PFI results
hct.save_object(obj=pfi, data_results="results", train_test="train", stage="pfi-cs-pre-smooth", sparam=0)

# Get predictions and accuracy on test data
test_pred = nn.predict(X = test_x)
test_acc = accuracy_score(y_true = test_y, y_pred = test_pred)

# Join the predictions and metrics in a dictionary (and save results)
test_res = {"preds": test_pred, "acc": test_acc}
hct.save_object(
  obj=test_res,
  data_results="results", 
  train_test="test", 
  stage="pred-and-metrics-cs-pre-smooth", 
  sparam=0
)
