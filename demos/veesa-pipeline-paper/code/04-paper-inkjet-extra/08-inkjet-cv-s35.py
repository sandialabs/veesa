#!/usr/bin/env python

# Contains: Alignment and jfPCA for cross-validation with inkjet data 
# (for one smoothing setting)
# Working Directory: The location of this script (code/inkjet-extra)

# Load packages
import fdasrsf as fs
import numpy as np
import pandas as pd
import pickle
import time
import veesa_functions as veesa

# Specify file paths
fp_data = "../../data/"
fp_res = "../../results/inkjet/"

# Load data
inkjet = pd.read_csv(fp_data + "inkjet-cleaned.csv")
inkjet_folds = pd.read_csv(fp_data + "inkjet-cv-folds.csv")

# Determine spectra
spectra = inkjet.spectra.unique()

# Determine the number of reps and folds
reps = inkjet_folds.rep.unique()
folds = inkjet_folds.fold.unique()

# Specify colors, number of box filter runs, and number of cores to use
colors = ['c', 'm', 'y']
s = 35
s_name = "35"
ncores = -1

# Start clock
start_time = time.time()

# Alignment and jfPCA within folds
for rep in reps:
  
  # Subset data to one rep
  inkjet_one_rep = inkjet_folds[inkjet_folds.rep == rep]
  inkjet_with_folds = pd.merge(
    inkjet_one_rep, 
    inkjet, 
    how = 'right', 
    on = ["printer", "sample"]
  )
  
  for fold in folds:
    
    # Split into train test
    train = inkjet_with_folds[inkjet_with_folds.fold != fold]
    test = inkjet_with_folds[inkjet_with_folds.fold == fold]
    
    for color in colors: 
      
      # Specify file path
      path = fp_res + "inkjet-cv" + "-s" + s_name + "-rep_" + str(rep) + "-fold_" + str(fold) + "-color_" + str(color)
      
      # Subset by a color
      train_color = train[train.color == color]
      train_color = train_color[["printer", "sample", "spectra", "intensity"]]
      test_color = test[test.color == color]
      test_color = test_color[["printer", "sample", "spectra", "intensity"]]    
      
      # Change to wide datasets
      train_color_wide = train_color.pivot(
        index=['printer','sample'],
        columns="spectra", 
        values="intensity"
      )
      test_color_wide = test_color.pivot(
        index=['printer','sample'],
        columns="spectra", 
        values="intensity"
      )
  
      # Convert to numpy arrays
      train_mat = np.array(train_color_wide.transpose())
      test_mat = np.array(test_color_wide.transpose())
      
      # Smooth data
      train_smooth = fs.smooth_data(f = train_mat, sparam = s)
      pickle.dump(train_smooth, open(path + "-smoothed-train.pkl", 'wb'))
      test_smooth = fs.smooth_data(f = test_mat, sparam = s)
      pickle.dump(test_smooth, open(path + "-smoothed-train.pkl", 'wb'))
      
      # Align training data
      train_aligned = fs.time_warping.fdawarp(train_smooth, spectra)
      train_aligned.srsf_align(parallel = True, cores = ncores, center = False, omethod = "DP")
      pickle.dump(train_aligned, open(path + "-aligned-train.pkl", 'wb'))
      
      # Apply jfPCA to training data
      train_jfpca = fs.fdajpca(train_aligned)
      train_jfpca.calc_fpca(
        no=train_aligned.time.shape[0],
        stds=np.array([-3.,-2.,-1.,0.,1.,2.,3.]),
        parallel=True,
        cores=ncores
      )
      pickle.dump(train_jfpca, open(path + "-jfpca-train.pkl", 'wb'))
      
      # Align and apply jfPCA to test data
      test_aligned_jfpca = veesa.prep_testing_data(
          f=test_smooth,
          time=spectra,
          aligned_train=train_aligned,
          fpca_train=train_jfpca,
          fpca_method="jfpca",
          omethod="DP"
        )
      pickle.dump(test_aligned_jfpca, open(path + "-aligned-jfpca-test.pkl", 'wb'))

# End clock
end_time = time.time()

# Compute duration and save
execution_time = end_time - start_time
veesa.save_object(
  obj=execution_time,
  folder_path=fp_res,
  train_test="train",
  stage="time-cv",
  sparam=s
)
