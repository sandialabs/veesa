#!/usr/bin/env python

# Preparation of training and testing H-CT data
# Author: Katherine Goode
# Date created: December 6, 2021

# Load packages
import pickle

# Load prepared H-CT data
hct_data = pickle.load(open("../data/hct-clean.pkl", "rb"))

# Extract the training/testing data
hct_train = hct_data[hct_data['dataset'] == "train"]
hct_test = hct_data[hct_data['dataset'] == "test"]

# Save training/testing data
pickle.dump(hct_train, open("../data/hct-train.pkl", 'wb'))
pickle.dump(hct_test, open("../data/hct-test.pkl", 'wb'))
