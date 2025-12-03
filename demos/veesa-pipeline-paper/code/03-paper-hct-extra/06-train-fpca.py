#!/usr/bin/env python

# Contains: Application of jfPCA, hfPCA, and vfPCA to H-CT data
# Working Directory: The location of this script (code/hct-extra)

# Load packages
import fdasrsf as fs
import numpy as np
import time
import veesa_functions as veesa

# Specify file paths
fp_data = "../../data/"
fp_res = "../../results/hct/"

# Specify smoothing parameters
sparam = [1, 5, 10, 15, 20, 25]

# Apply jfPCA/hfPCA/vfPCA for each of the smoothing parameters
for s in sparam:

  # Print sparam to keep track of progress
  print("Working on sparam:" + str(s))

  # Load aligned data
  train_aligned = veesa.load_object(
    folder_path=fp_res,
    train_test="train", 
    stage="aligned", 
    sparam=s
  )
  
  ### jfPCA ----------------------------------------------------------------
  
  # Start clock
  start_time = time.time()

  # Apply jfPCA
  train_jfpca = fs.fdajpca(train_aligned)
  train_jfpca.calc_fpca(
    no=train_aligned.time.shape[0], 
    stds=np.array([-3.,-2.,-1.,0.,1.,2.,3.]),
    parallel=True, 
    cores=-1
  )
  
  # End clock
  end_time = time.time()
  
  # Compute duration and save
  execution_time = end_time - start_time
  veesa.save_object(
    obj=execution_time,
    folder_path=fp_res,
    train_test="train",
    stage="time-jfpca",
    sparam=s
  )

  # Save jfPCA results
  veesa.save_object(
    obj=train_jfpca,
    folder_path=fp_res, 
    train_test="train", 
    stage="jfpca", 
    sparam=s
  )
  veesa.save_object(
    obj=train_jfpca.f_pca, 
    folder_path=fp_res, 
    train_test="train", 
    stage="jfpca-pc-dirs", 
    sparam=s
  )
  veesa.save_object(
    obj=train_jfpca.latent, 
    folder_path=fp_res, 
    train_test="train", 
    stage="jfpca-latent", 
    sparam=s
  )

  # Compute and save aligned versions of jfPCA principal directions
  train_jfpca_aligned = veesa.align_pcdirs(
    aligned_train=train_aligned, 
    jfpca_train=train_jfpca
  )
  veesa.save_object(
    obj=train_jfpca_aligned, 
    folder_path=fp_res,
    train_test="train", 
    stage="jfpca-centered", 
    sparam=s
  )
  veesa.save_object(
    obj=train_jfpca_aligned.f_pca, 
    folder_path=fp_res, 
    train_test="train", 
    stage="jfpca-centered-pc-dirs", 
    sparam=s
  )
  veesa.save_object(
    obj=train_jfpca_aligned.latent, 
    folder_path=fp_res, 
    train_test="train", 
    stage="jfpca-centered-latent", 
    sparam=s
  )
  
  ### vfPCA ----------------------------------------------------------------
  
  # Start clock
  start_time = time.time()
  
  # Apply vfPCA
  train_vfpca = fs.fdavpca(train_aligned)
  train_vfpca.calc_fpca(
    no=train_aligned.time.shape[0], 
    stds=np.array([-3.,-2.,-1.,0.,1.,2.,3.])
  )

  # End clock
  end_time = time.time()
  
  # Compute duration and save
  execution_time = end_time - start_time
  veesa.save_object(
    obj=execution_time,
    folder_path=fp_res,
    train_test="train",
    stage="time-vfpca",
    sparam=s
  )
  
  # Save vfPCA results
  veesa.save_object(
    obj=train_vfpca,
    folder_path=fp_res,
    train_test="train", 
    stage="vfpca", 
    sparam=s
  )
  veesa.save_object(
    obj=train_vfpca.f_pca, 
    folder_path=fp_res, 
    train_test="train", 
    stage="vfpca-pc-dirs",
    sparam=s
  )
  veesa.save_object(
    obj=train_vfpca.latent, 
    folder_path=fp_res, 
    train_test="train", 
    stage="vfpca-latent", 
    sparam=s
  )
  
  ### hfPCA ----------------------------------------------------------------
  
  # Start clock
  start_time = time.time()
  
  # Apply hfPCA
  train_hfpca = fs.fdahpca(train_aligned)
  train_hfpca.calc_fpca(
    no=train_aligned.time.shape[0], 
    stds=np.array([-3.,-2.,-1.,0.,1.,2.,3.])
  )

  # End clock
  end_time = time.time()
  
  # Compute duration and save
  execution_time = end_time - start_time
  veesa.save_object(
    obj=execution_time,
    folder_path=fp_res,
    train_test="train",
    stage="time-hfpca",
    sparam=s
  )
  
  # Save hfPCA results
  veesa.save_object(
    obj=train_hfpca,
    folder_path=fp_res,
    train_test="train", 
    stage="hfpca", 
    sparam=s
  )
  veesa.save_object(
    obj=train_hfpca.gam_pca, 
    folder_path=fp_res, 
    train_test="train", 
    stage="hfpca-pc-dirs", 
    sparam=s
  )
  veesa.save_object(
    obj=train_hfpca.latent, 
    folder_path=fp_res, 
    train_test="train", 
    stage="hfpca-latent", 
    sparam=s
  )
