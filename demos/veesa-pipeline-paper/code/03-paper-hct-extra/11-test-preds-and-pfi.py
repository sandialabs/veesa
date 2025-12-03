#!/usr/bin/env python

# Contains: Application of PFI to models using ESA fPCA results (with test data)
# Working Directory: The location of this script (code/hct-extra)

# Load packages
import pickle
import veesa_functions as veesa
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
import time

# Specify file paths
fp_data = "../../data/"
fp_res = "../../results/hct/"

# Load H-CT data
test = pickle.load(open(fp_data + "hct-test.pkl", "rb"))

# Prepare the response variables for training
test_y = test[["id","material"]].drop_duplicates()["material"]

# Specify smoothing parameters
sparam = [1, 5, 10, 15, 20, 25]

# Compute PFI
for s in sparam:
    
  # Print sparam to keep track of progress
  print("Working on sparam:" + str(s))
  
  ### jfPCA ----------------------------------------------------------------
  
  # Load fPCA results
  test_aligned_jfpca = veesa.load_object(
    folder_path=fp_res,
    train_test="test", 
    stage="aligned-jfpca", 
    sparam=s
  )
  
  # Load model
  nn_jfpca = veesa.load_object(
    folder_path=fp_res, 
    train_test="train",
    stage="nn-jfpca",
    sparam=s
  )
  
  # Start clock
  start_time = time.time()
  
  # Get predictions and accuracy on test data and join
  test_pred_jfpca = nn_jfpca.predict(X = test_aligned_jfpca["coef"])
  test_acc_jfpca = accuracy_score(y_true = test_y, y_pred = test_pred_jfpca)
  test_res_jfpca = {"preds": test_pred_jfpca, "acc": test_acc_jfpca}
  
  # End clock
  end_time = time.time()
  
  # Compute duration and save
  execution_time = end_time - start_time
  veesa.save_object(
    obj=execution_time,
    folder_path=fp_res,
    train_test="test",
    stage="time-pred-and-metrics-jfpca",
    sparam=s
  )
  
  # Save test data results
  veesa.save_object(
    obj=test_res_jfpca,
    folder_path=fp_res, 
    train_test="test", 
    stage="pred-and-metrics-jfpca", 
    sparam=s
  )
  
  # Start clock
  start_time = time.time()
  
  # Compute PFI for the neural network using accuracy
  pfi_jfpca = permutation_importance(
    estimator=nn_jfpca, 
    X=test_aligned_jfpca["coef"], 
    y=test_y,
    scoring="accuracy", 
    n_repeats=5, 
    n_jobs=5, 
    random_state=20211213
  )
  
  # End clock
  end_time = time.time()
  
  # Compute duration and save
  execution_time = end_time - start_time
  veesa.save_object(
    obj=execution_time,
    folder_path=fp_res,
    train_test="test",
    stage="time-pfi-jfpca",
    sparam=s
  )
  
  # Save PFI
  veesa.save_object(
    obj=pfi_jfpca, 
    folder_path=fp_res, 
    train_test="test", 
    stage="pfi-jfpca", 
    sparam=s
  )
  
  ### vfPCA ----------------------------------------------------------------
  
  # Load fPCA results
  test_aligned_vfpca = veesa.load_object(
    folder_path=fp_res, 
    train_test="test", 
    stage="aligned-vfpca", 
    sparam=s
  )
  
  # Load model
  nn_vfpca = veesa.load_object(
    folder_path=fp_res, 
    train_test="train",
    stage="nn-vfpca", 
    sparam=s
  )
  
  # Start clock
  start_time = time.time()
  
  # Get predictions and accuracy on test data and join
  test_pred_vfpca = nn_vfpca.predict(X = test_aligned_vfpca["coef"])
  test_acc_vfpca = accuracy_score(y_true = test_y, y_pred = test_pred_vfpca)
  test_res_vfpca = {"preds": test_pred_vfpca, "acc": test_acc_vfpca}
  
  # End clock
  end_time = time.time()
  
  # Compute duration and save
  execution_time = end_time - start_time
  veesa.save_object(
    obj=execution_time,
    folder_path=fp_res,
    train_test="test",
    stage="time-pred-and-metrics-vfpca",
    sparam=s
  )
  
  # Save test data results
  veesa.save_object(
    obj=test_res_vfpca,
    folder_path=fp_res, 
    train_test="test", 
    stage="pred-and-metrics-vfpca", 
    sparam=s
  )
  
  # Start clock
  start_time = time.time()
  
  # Compute PFI for the neural network using accuracy
  pfi_vfpca = permutation_importance(
    estimator=nn_vfpca, 
    X=test_aligned_vfpca["coef"], 
    y=test_y,
    scoring="accuracy", 
    n_repeats=5, 
    n_jobs=5, 
    random_state=20211213
  )
  
  # End clock
  end_time = time.time()
  
  # Compute duration and save
  execution_time = end_time - start_time
  veesa.save_object(
    obj=execution_time,
    folder_path=fp_res,
    train_test="test",
    stage="time-pfi-vfpca",
    sparam=s
  )
  
  # Save PFI
  veesa.save_object(
    obj=pfi_vfpca, 
    folder_path=fp_res, 
    train_test="test", 
    stage="pfi-vfpca", 
    sparam=s
  )
  
  # Start clock
  start_time = time.time()
  
  # Get predictions and accuracy on test data and join
  test_pred_vfpca = nn_vfpca.predict(X = test_aligned_vfpca["coef"])
  test_acc_vfpca = accuracy_score(y_true = test_y, y_pred = test_pred_vfpca)
  test_res_vfpca = {"preds": test_pred_vfpca, "acc": test_acc_vfpca}
  
  # End clock
  end_time = time.time()
  
  # Compute duration and save
  execution_time = end_time - start_time
  veesa.save_object(
    obj=execution_time,
    folder_path=fp_res,
    train_test="test",
    stage="time-pred-and-metrics-vfpca",
    sparam=s
  )
  
  # Save test data results
  veesa.save_object(
    obj=test_res_vfpca,
    folder_path=fp_res, 
    train_test="test", 
    stage="pred-and-metrics-vfpca", 
    sparam=s
  )
  
  ### hfPCA ----------------------------------------------------------------
  
  # Load ffPCA results
  test_aligned_hfpca = veesa.load_object(
    folder_path=fp_res, 
    train_test="test", 
    stage="aligned-hfpca", 
    sparam=s
  )

  # Load model
  nn_hfpca = veesa.load_object(
    folder_path=fp_res, 
    train_test="train", 
    stage="nn-hfpca", 
    sparam=s
  )
  
  # Start clock
  start_time = time.time()
  
  # Get predictions and accuracy on test data and join
  test_pred_hfpca = nn_hfpca.predict(X = test_aligned_hfpca["coef"])
  test_acc_hfpca = accuracy_score(y_true = test_y, y_pred = test_pred_hfpca)
  test_res_hfpca = {"preds": test_pred_hfpca, "acc": test_acc_hfpca}
  
  # End clock
  end_time = time.time()
  
  # Compute duration and save
  execution_time = end_time - start_time
  veesa.save_object(
    obj=execution_time,
    folder_path=fp_res,
    train_test="test",
    stage="time-pred-and-metrics-hfpca",
    sparam=s
  )
  
  # Save test data results
  veesa.save_object(
    obj=test_res_hfpca,
    folder_path=fp_res, 
    train_test="test", 
    stage="pred-and-metrics-hfpca", 
    sparam=s
  )
  
  # Start clock
  start_time = time.time()
  
  # Compute PFI for the neural network using accuracy
  pfi_hfpca = permutation_importance(
    estimator=nn_hfpca, 
    X=test_aligned_hfpca["coef"], 
    y=test_y,
    scoring="accuracy", 
    n_repeats=5, 
    n_jobs=5, 
    random_state=20211213
  )
  
  # End clock
  end_time = time.time()
  
  # Compute duration and save
  execution_time = end_time - start_time
  veesa.save_object(
    obj=execution_time,
    folder_path=fp_res,
    train_test="test",
    stage="time-pfi-hfpca",
    sparam=s
  )
  
  # Save PFI
  veesa.save_object(
    obj=pfi_hfpca, 
    folder_path=fp_res,
    train_test="test", 
    stage="pfi-hfpca", 
    sparam=s
  )

  # Start clock
  start_time = time.time()
  
  # Get predictions and accuracy on test data and join
  test_pred_hfpca = nn_hfpca.predict(X = test_aligned_hfpca["coef"])
  test_acc_hfpca = accuracy_score(y_true = test_y, y_pred = test_pred_hfpca)
  test_res_hfpca = {"preds": test_pred_hfpca, "acc": test_acc_hfpca}
  
  # End clock
  end_time = time.time()
  
  # Compute duration and save
  execution_time = end_time - start_time
  veesa.save_object(
    obj=execution_time,
    folder_path=fp_res,
    train_test="test",
    stage="time-pred-and-metrics-hfpca",
    sparam=s
  )
  
  # Save test data results
  veesa.save_object(
    obj=test_res_hfpca,
    folder_path=fp_res, 
    train_test="test", 
    stage="pred-and-metrics-hfpca", 
    sparam=s
  )
