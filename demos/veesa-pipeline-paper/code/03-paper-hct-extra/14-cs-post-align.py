#!/usr/bin/env python

# Contains: Cross-sectional approach (post-alignment) to modeling H-CT data
# Working Directory: The location of this script (code/hct-extra)

# Load packages
import pickle
import veesa_functions as veesa
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

# Specify file paths
fp_data = "../../data/"
fp_res = "../../results/hct/"

# Load training and testing data
train = pickle.load(open(fp_data + "hct-train.pkl", "rb"))
test = pickle.load(open(fp_data + "hct-test.pkl", "rb"))

# Prepare response variables
train_y = train[["id","material"]].drop_duplicates()["material"]
test_y = test[["id","material"]].drop_duplicates()["material"]

# Specify smoothing parameters
sparam = [1, 5, 10, 15, 20, 25]

# Apply CS approach for each smoothing parameter
for s in sparam:

  # Load the smoothed train/test data
  train_x = veesa.load_object(
    folder_path=fp_res, 
    train_test="train", 
    stage="aligned", sparam=s
  ).fn.T
  test_x = veesa.load_object(
    folder_path=fp_res, 
    train_test="test", 
    stage="aligned-jfpca", 
    sparam=s
  )["fn"].T

  # Train neural network, compute performance metrics, and save results
  veesa.apply_model(
    x=train_x, 
    y=train_y,
    analysis_name="cs-post-align",
    sparam=s, 
    folder_path=fp_res,
    seed=20211213
  )

  # Load the model
  nn = veesa.load_object(
    folder_path=fp_res,
    train_test="train", 
    stage="nn-cs-post-align", 
    sparam=s
  )

  # Apply PFI
  pfi = permutation_importance(
    estimator=nn,
    X=train_x,
    y=train_y,
    scoring="accuracy",
    n_repeats=5,
    n_jobs=5,
    random_state=20211213
  )

  # Save PFI results
  veesa.save_object(
    obj=pfi, 
    folder_path=fp_res, 
    train_test="train", 
    stage="pfi-cs-post-align", 
    sparam=s
  )

  # Get predictions and accuracy on test data
  test_pred = nn.predict(X = test_x)
  test_acc = accuracy_score(y_true = test_y, y_pred = test_pred)

  # Join the predictions and metrics in a dictionary (and save results)
  test_res = {"preds": test_pred, "acc": test_acc}
  veesa.save_object(
    obj=test_res,
    folder_path=fp_res,
    train_test="test",
    stage="pred-and-metrics-cs-post-align",
    sparam=s
  )
