#!/usr/bin/env python

# Application of jfPCA, hfPCA, and vfPCA to H-CT data
# Author: Katherine Goode
# Date created: December 13, 2021

# Load packages
import fdasrsf as fs
import hct_functions as hct
import numpy as np
import pickle

# Specify smoothing parameters
sparam = [1, 5, 10, 15, 20, 25]

# Apply jfPCA/hfPCA/vfPCA for each of the smoothing parameters
for s in sparam:

  # Print sparam to keep track of progress
  print("Working on sparam:" + str(s))

  # Load aligned data
  train_aligned = hct.load_object(data_results = "data", train_test="train", stage="aligned", sparam=s)
  
  # Apply jfPCA
  train_jfpca = fs.fdajpca(train_aligned)
  train_jfpca.calc_fpca(no=train_aligned.time.shape[0], stds=np.array([-3.,-2.,-1.,0.,1.,2.,3.]), parallel=True, cores=-1)

  # Save jfPCA results
  hct.save_object(obj=train_jfpca, data_results="data", train_test="train", stage="jfpca", sparam=s)
  hct.save_object(obj=train_jfpca.f_pca, data_results="data", train_test="train", stage="jfpca-pc-dirs", sparam=s)
  hct.save_object(obj=train_jfpca.latent, data_results="data", train_test="train", stage="jfpca-latent", sparam=s)

  # Compute and save aligned versions of jfPCA principal directions
  train_jfpca = hct.load_object(data_results = "data", train_test="train", stage="jfpca", sparam=s)
  train_jfpca_aligned = hct.align_pcdirs(aligned_train=train_aligned, jfpca_train=train_jfpca)
  hct.save_object(obj=train_jfpca_aligned, data_results="data", train_test="train", stage="jfpca-centered", sparam=s)
  hct.save_object(obj=train_jfpca_aligned.f_pca, data_results="data", train_test="train", stage="jfpca-centered-pc-dirs", sparam=s)
  hct.save_object(obj=train_jfpca_aligned.latent, data_results="data", train_test="train", stage="jfpca-centered-latent", sparam=s)
  
  # Apply vfPCA
  train_vfpca = fs.fdavpca(train_aligned)
  train_vfpca.calc_fpca(no=train_aligned.time.shape[0], stds=np.array([-3.,-2.,-1.,0.,1.,2.,3.]))

  # Save vfPCA results
  hct.save_object(obj=train_vfpca, data_results="data", train_test="train", stage="vfpca", sparam=s)
  hct.save_object(obj=train_vfpca.f_pca, data_results="data", train_test="train", stage="vfpca-pc-dirs", sparam=s)
  hct.save_object(obj=train_vfpca.latent, data_results="data", train_test="train", stage="vfpca-latent", sparam=s)
  
  # Apply hfPCA
  train_hfpca = fs.fdahpca(train_aligned)
  train_hfpca.calc_fpca(no=train_aligned.time.shape[0], stds=np.array([-3.,-2.,-1.,0.,1.,2.,3.]))

  # Save hfPCA results
  hct.save_object(obj=train_hfpca, data_results="data", train_test="train", stage="hfpca", sparam=s)
  hct.save_object(obj=train_hfpca.gam_pca, data_results="data", train_test="train", stage="hfpca-pc-dirs", sparam=s)
  hct.save_object(obj=train_hfpca.latent, data_results="data", train_test="train", stage="hfpca-latent", sparam=s)
  
