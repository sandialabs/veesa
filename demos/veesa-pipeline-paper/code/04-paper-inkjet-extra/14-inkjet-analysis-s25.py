#!/usr/bin/env python

# Contains: Alignment and jfPCA for all inkjet data (for one smoothing setting)
# Working Directory: The location of this script (code/inkjet-extra)

# Load packages
import fdasrsf as fs
import numpy as np
import pandas as pd
import pickle
import time

# Specify number of cores to use
ncores = -1

# Specify number of times to run box filter
s = 25

# Specify file paths
fp_data = "../../data/"
fp_res = "../../results/inkjet/"

#### Data Steps ---------------------------------------------------------------

# Load data
spectra = pd.read_csv(fp_data + "inkjet-spectra.csv")
cyan = pd.read_csv(fp_data + "inkjet-matrix-cyan.csv")
magenta = pd.read_csv(fp_data + "inkjet-matrix-magenta.csv")
yellow = pd.read_csv(fp_data + "inkjet-matrix-yellow.csv")

# Extract spectra values
spectra = np.array(spectra.spectra)

# Convert to numpy arrays
cyan = np.array(cyan)
magenta = np.array(magenta)
yellow = np.array(yellow)

#### Analysis of Cyan ---------------------------------------------------------

# Smooth functions
start_time = time.time()
cyan_smoothed = fs.smooth_data(f=cyan, sparam=s)
end_time = time.time()
execution_time = end_time - start_time
pickle.dump(execution_time, open(fp_res + "inkjet-time-s25-smoothed-cyan.pkl", "wb"))
pickle.dump(cyan_smoothed, open(fp_res + "inkjet-s25-smoothed-cyan.pkl", "wb"))

# Align data
start_time = time.time()
cyan_aligned = fs.time_warping.fdawarp(cyan_smoothed, spectra)
cyan_aligned.srsf_align(parallel=True, cores=ncores, center=True, omethod="DP")
end_time = time.time()
execution_time = end_time - start_time
pickle.dump(execution_time, open(fp_res + "inkjet-time-s25-aligned-cyan.pkl", "wb"))
pickle.dump(cyan_aligned, open(fp_res + "inkjet-s25-aligned-cyan.pkl", "wb"))

# Apply jfPCA
start_time = time.time()
cyan_jfpca = fs.fdajpca(cyan_aligned)
cyan_jfpca.calc_fpca(no=cyan_aligned.time.shape[0], stds=np.array([-3.,-2.,-1.,0.,1.,2.,3.]), parallel=True, cores=-1)
end_time = time.time()
execution_time = end_time - start_time
pickle.dump(execution_time, open(fp_res + "inkjet-time-s25-jfpca-cyan.pkl", "wb"))
pickle.dump(cyan_jfpca, open(fp_res + "inkjet-s25-jfpca-cyan.pkl", "wb"))

#### Analysis of Magenta ------------------------------------------------------

# Smooth functions
start_time = time.time()
magenta_smoothed = fs.smooth_data(f=magenta, sparam=s)
end_time = time.time()
execution_time = end_time - start_time
pickle.dump(execution_time, open(fp_res + "inkjet-time-s25-smoothed-magenta.pkl", "wb"))
pickle.dump(magenta_smoothed, open(fp_res + "inkjet-s25-smoothed-magenta.pkl", "wb"))

# Align data
start_time = time.time()
magenta_aligned = fs.time_warping.fdawarp(magenta_smoothed, spectra)
magenta_aligned.srsf_align(parallel=True, cores=ncores, center=True, omethod="DP")
end_time = time.time()
execution_time = end_time - start_time
pickle.dump(execution_time, open(fp_res + "inkjet-time-s25-aligned-magenta.pkl", "wb"))
pickle.dump(magenta_aligned, open(fp_res + "inkjet-s25-aligned-magenta.pkl", "wb"))

# Apply jfPCA
start_time = time.time()
magenta_jfpca = fs.fdajpca(magenta_aligned)
magenta_jfpca.calc_fpca(no=magenta_aligned.time.shape[0], stds=np.array([-3.,-2.,-1.,0.,1.,2.,3.]), parallel=True, cores=-1)
end_time = time.time()
execution_time = end_time - start_time
pickle.dump(execution_time, open(fp_res + "inkjet-time-s25-jfpca-magenta.pkl", "wb"))
pickle.dump(magenta_jfpca, open(fp_res + "inkjet-s25-jfpca-magenta.pkl", "wb"))

#### Analysis of Yellow --------------------------------------------------------

# Smooth functions
start_time = time.time()
yellow_smoothed = fs.smooth_data(f=yellow, sparam=s)
end_time = time.time()
execution_time = end_time - start_time
pickle.dump(execution_time, open(fp_res + "inkjet-time-s25-smoothed-yellow.pkl", "wb"))
pickle.dump(yellow_smoothed, open(fp_res + "inkjet-s25-smoothed-yellow.pkl", "wb"))

# Align data
start_time = time.time()
yellow_aligned = fs.time_warping.fdawarp(yellow_smoothed, spectra)
yellow_aligned.srsf_align(parallel=True, cores=ncores, center=True, omethod="DP")
end_time = time.time()
execution_time = end_time - start_time
pickle.dump(execution_time, open(fp_res + "inkjet-time-s25-aligned-yellow.pkl", "wb"))
pickle.dump(yellow_aligned, open(fp_res + "inkjet-s25-aligned-yellow.pkl", "wb"))

# Apply jfPCA
start_time = time.time()
yellow_jfpca = fs.fdajpca(yellow_aligned)
yellow_jfpca.calc_fpca(no=yellow_aligned.time.shape[0], stds=np.array([-3.,-2.,-1.,0.,1.,2.,3.]), parallel=True, cores=-1)
end_time = time.time()
execution_time = end_time - start_time
pickle.dump(execution_time, open(fp_res + "inkjet-time-s25-jfpca-yellow.pkl", "wb"))
pickle.dump(yellow_jfpca, open(fp_res + "inkjet-s25-jfpca-yellow.pkl", "wb"))
