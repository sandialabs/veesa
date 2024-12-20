#!/usr/bin/env python

# VEESA pipeline applied to full dataset
# Author: Katherine Goode
# Date created: December 16, 2022

# Load packages
import fdasrsf as fs
import numpy as np
import pandas as pd
import pickle

# Specify number of cores to use
ncores = 40

# Specify number of times to run box filter
s = 30

#### Data Steps ---------------------------------------------------------------

# Load data
spectra = pd.read_csv('../data/inkjet-spectra.csv')
cyan = pd.read_csv('../data/inkjet-matrix-cyan.csv')
magenta = pd.read_csv('../data/inkjet-matrix-magenta.csv')
yellow = pd.read_csv('../data/inkjet-matrix-yellow.csv')

# Extract spectra values
spectra = np.array(spectra.spectra)

# Convert to numpy arrays
cyan = np.array(cyan)
magenta = np.array(magenta)
yellow = np.array(yellow)

#### Analysis of Cyan ---------------------------------------------------------

# Smooth functions
cyan_smoothed = fs.smooth_data(f=cyan, sparam=s)
pickle.dump(cyan_smoothed, open("../data/inkjet-s30-smoothed-cyan.pkl", 'wb'))

# Align data
cyan_aligned = fs.time_warping.fdawarp(cyan_smoothed, spectra)
cyan_aligned.srsf_align(parallel=True, cores=ncores, center=True, omethod="DP")
pickle.dump(cyan_aligned, open("../data/inkjet-s30-aligned-cyan.pkl", 'wb'))

# Apply jfPCA
cyan_jfpca = fs.fdajpca(cyan_aligned)
cyan_jfpca.calc_fpca(no=cyan_aligned.time.shape[0], stds=np.array([-3.,-2.,-1.,0.,1.,2.,3.]), parallel=True, cores=-1)
pickle.dump(cyan_jfpca, open("../data/inkjet-s30-jfpca-cyan.pkl", 'wb'))

#### Analysis of Magenta ------------------------------------------------------

# Smooth functions
magenta_smoothed = fs.smooth_data(f=magenta, sparam=s)
pickle.dump(magenta_smoothed, open("../data/inkjet-s30-smoothed-magenta.pkl", 'wb'))

# Align data
magenta_aligned = fs.time_warping.fdawarp(magenta_smoothed, spectra)
magenta_aligned.srsf_align(parallel=True, cores=ncores, center=True, omethod="DP")
pickle.dump(magenta_aligned, open("../data/inkjet-s30-aligned-magenta.pkl", 'wb'))

# Apply jfPCA
magenta_jfpca = fs.fdajpca(magenta_aligned)
magenta_jfpca.calc_fpca(no=magenta_aligned.time.shape[0], stds=np.array([-3.,-2.,-1.,0.,1.,2.,3.]), parallel=True, cores=-1)
pickle.dump(magenta_jfpca, open("../data/inkjet-s30-jfpca-magenta.pkl", 'wb'))

#### Analysis of Yellow --------------------------------------------------------

# Smooth functions
yellow_smoothed = fs.smooth_data(f=yellow, sparam=s)
pickle.dump(yellow_smoothed, open("../data/inkjet-s30-smoothed-yellow.pkl", 'wb'))

# Align data
yellow_aligned = fs.time_warping.fdawarp(yellow_smoothed, spectra)
yellow_aligned.srsf_align(parallel=True, cores=ncores, center=True, omethod="DP")
pickle.dump(yellow_aligned, open("../data/inkjet-s30-aligned-yellow.pkl", 'wb'))

# Apply jfPCA
yellow_jfpca = fs.fdajpca(yellow_aligned)
yellow_jfpca.calc_fpca(no=yellow_aligned.time.shape[0], stds=np.array([-3.,-2.,-1.,0.,1.,2.,3.]), parallel=True, cores=-1)
pickle.dump(yellow_jfpca, open("../data/inkjet-s30-jfpca-yellow.pkl", 'wb'))
