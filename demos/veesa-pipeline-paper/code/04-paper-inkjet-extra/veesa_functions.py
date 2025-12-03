def save_object(obj, folder_path, train_test, stage, sparam, sub=False):
  
  """ Apply smoothing to H-CT data  
    Args:
        obj: object to be saved
        folder_path: character string indicating which folder to store the object in 
        train_test: character string indicating whether object is associated with 'train' or 'test' data
        stage: character string indicating the stage in the analysis (such as "smoothed" or "aligned")
        sparam: smoothing parameter (From fdasrsf: "Number of times to run box filter")
        sub: indicates whether the full data is used (True) or a subset (False)
        
    Returns: Saved pickle file
  """
  
  # Load packages
  import pickle
  
  # Create sparam character (for file name)
  if sparam < 10:
    sparam_name = "0" + str(sparam)
  else:
    sparam_name = sparam

  # Save object
  if sub == False:
    fp = folder_path + "hct-" + train_test + "-" + stage + "-sparam" + str(sparam_name)  + ".pkl"
    pickle.dump(obj, open(fp, 'wb'))
  else: 
    fp = folder_path + "hct-sub-" + train_test + "-" + stage + "-sparam" + str(sparam_name)  + ".pkl"
    pickle.dump(obj, open(fp, 'wb'))
  
def load_object(folder_path, train_test, stage, sparam):
  
  """ Load smoothed H-CT data
    Args:
        folder_path: character string indicating which folder to store the object in
        train_test: character string indicating whether the data is training or testing data ('train' or 'test')
        stage: character string indicating the stage in the analysis (such as "smoothed" or "aligned")
        sparam: smoothing parameter (From fdasrsf: "Number of times to run box filter")
        
    Returns: Loaded pickle file
  """
  
  # Load packages
  import pickle
  
  # Create sparam character (for file name)
  if sparam < 10:
    sparam_name = "0" + str(sparam)
  else:
    sparam_name = sparam
    
  # Load object
  fp = folder_path + "hct-" + train_test + "-" + stage + "-sparam" + str(sparam_name)  + ".pkl"
  obj = pickle.load(open(fp, "rb"))
  return(obj)
  
def apply_model(x, y, analysis_name, sparam, folder_path, seed):
  
  """ Train neural network and compute predictions and model metrics 
    Args:
        x: NxM matrix of functions (M=number of samples per function, N=number of functions)
        y: Response variable
        analysis_name: Name of analysis to use for labelling saved files
        sparam: Smoothing parameter used
        folder_path: character string indicating which folder to store the object in
        seed: Seed to use for any analyses with randomness
        
    Returns:
        Saves two files:
            1. Neural network
            2. Predictions and model metrics
  """
  
  # Load packages
  import pickle
  from sklearn.metrics import accuracy_score
  from sklearn.neural_network import MLPClassifier

  # Train neural network
  nn = MLPClassifier(random_state = seed)
  nn.fit(X = x, y = y)
  
  # Save neural network
  save_object(obj=nn, folder_path=folder_path, train_test="train", stage="nn-"+analysis_name, sparam=sparam)
  
  # Compute predictions on training data
  nn_pred = nn.predict(X = x)

  # Compute performance metrics on training data
  nn_acc = accuracy_score(y_true = y, y_pred = nn_pred)

  # Join the predictions and metrics in a dictionary
  nn_res = {
    "preds": nn_pred,
    "acc": nn_acc
  }

  # Save predictions, and performance metrics
  save_object(obj=nn_res, folder_path=folder_path, train_test="train", stage="pred-and-metrics-"+analysis_name, sparam=sparam)
  
def prep_testing_data(f, time, aligned_train, fpca_train, fpca_method, omethod):
  
  """ Function for preparing test data corresponding to ESA alignment and fPCA (joint, 
  horizontal, or vertical) applied to training data
    Args:
        f: test data matrix (N x M) of M functions with N samples
        time: vector of size N describing the sample points
        aligned_train: object returned from applying time warping to training data
        fpca_train: object returned from applying fPCA (joint, horiztonal, or vertical) to training data
        fpca_method: string specifying the type of fPCA used ('jfpca', 'hfpca', or 'vfpca')
        omethod: method used for optimization when computing the Karcher mean (DP,DP2,RBFGS,DPo)
        
    Returns:
        Dictionary containing (varies slightly based on fpca method used):
          - time: vector of times when functions are observed (length of N)
          - f0: original test data functions - matrix (N x M) of M functions with N samples
          - fn: aligned test data functions - similar structure to f0
          - q0: original test data SRSFs - similar structure to f0
          - qn: aligned test data SRSFs - similar structure to f0
          - mqn: training data SRSF mean (test data functions are aligned to this function)
          - gam: test data warping functions - similar structure to f0
          - coef: test data principal component coefficients
          - psi: test data warping function SRVFs - matrix (N x M) of M functions with N samples (jfpca and hfpca only)
          - nu: test data shooting functions - matrix (N x M) of M functions with N samples (jfpca and hfpca only)
          - g: test data combination of aligned and shooting functions (jfpca only)
  """
  
  # Load packages
  import fdasrsf as fs
  import numpy as np
  import pandas

  # Determine the number of functions in the test data
  M = f.shape[1]
  
  # Change times to be between 0 and 1
  time = np.linspace(start=0, stop=1, num=len(time)).astype(float)
  
  # Identify Karcher mean training data SRSFs
  q_mean_train = aligned_train.mqn
  
  # Align functions to mqn (ignoring centering) and compute items needed for fPCA
  q = np.zeros(f.shape)
  gam = np.zeros(f.shape)
  fn = np.zeros(f.shape)
  qn = np.zeros(f.shape)
  if fpca_method == "jfpca" or fpca_method == "hfpca":
    psi = np.zeros(f.shape)
    binsize = np.mean(np.diff(time))
  for m in range(M):
    # Convert the test data to SRSFs
    q[:,m] = fs.f_to_srsf(f=f[:,m], time=time)
    # Obtain warping functions needed to align test data to training data Karcher Mean
    gam[:,m] = fs.optimum_reparam(q1=q_mean_train, time=time, q2=q[:,m], method = omethod)
    # Apply warping functions to align test data functions
    fn[:,m] = fs.warp_f_gamma(time=time, f=f[:,m], gam=gam[:,m])
    # Compute the SRSFs of the aligned functions
    qn[:,m] = fs.f_to_srsf(f=fn[:,m], time=time)
    if fpca_method == "jfpca" or fpca_method == "hfpca":
      # Compute SRSFs of test data warping functions:
      psi[:,m] = np.sqrt(np.gradient(gam[:,m], binsize))
  
  # Compute shooting vectors (if needed)  
  if fpca_method == "jfpca" or fpca_method == "hfpca": 
    nu = np.zeros(f.shape)  
    if fpca_method == "jfpca":
      mu_psi = fpca_train.mu_psi
    elif fpca_method == "hfpca":
      mu_psi = psi.mean(axis=1)
    for m in range(M):
      # Compute test data shooting functions:
      nu[:,m], theta = fs.inv_exp_map(Psi=mu_psi, psi=psi[:,m]) 
  
  # Obtain id values (if needed)
  if fpca_method == "jfpca" or fpca_method == "vfpca": 
    f_id = fn[fpca_train.id, :]
    q_id = np.sign(f_id) * np.sqrt(np.abs(f_id))
  
  # Compute the principal components for the test data
  pcs = np.zeros(f.T.shape)
  if fpca_method == "jfpca":
    # Create the vector g with aligned functions and shooting vectors
    nu_scaled = nu * fpca_train.C
    g = np.vstack((qn, q_id, nu_scaled))
    # Now compute the PCs
    for i in range(pcs.shape[0]):
      g_centered = g[:,i] - fpca_train.mu_g
      for j in range(pcs.shape[1]): 
        pcs[i,j] = np.dot(g_centered, fpca_train.U[:,j])
  elif fpca_method == "vfpca":
    h = np.vstack((qn, q_id))
    for i in range(pcs.shape[0]):
      h_centered = h[:,i] - np.hstack((q_mean_train, np.mean(q_id)))
      for j in range(pcs.shape[1]):
        pcs[i,j] = np.dot(h_centered, fpca_train.U[:,j])
  elif fpca_method == "hfpca":
    nu_mean = nu.mean(axis=1)
    for i in range(pcs.shape[0]):
      nu_centered = nu[:,i] - nu_mean
      for j in range(pcs.shape[1]): 
        pcs[i,j] = np.dot(nu_centered, fpca_train.U[:,j])
  
  # Return the results from alignment and fPCA
  res = {
    'time': time, 
    'f0': f, 
    'fn': fn,
    'q0': q, 
    'qn': qn, 
    'mqn': q_mean_train,
    'gam': gam,
    'coef': pcs
    }
  if fpca_method == "hfpca" or fpca_method == "jfpca":
    res['psi'] = psi
    res['nu'] = nu
  if fpca_method == "jfpca":
    res['g'] = g 
  return(res)

def align_pcdirs(aligned_train, jfpca_train):
  
  """ Function for aligning principal directions obtained from 'fdajpca' with
  uncentered warping functions. By not centering the warping functions, an 
  alignment of the principal directions is needed for best interpretation.
  
  Args:
    aligned_train: object output from 'time_warping'
    jfpca_train: object output from 'fdajpca'
  
  Returns:
    Object with the same structure as output from 'fdajpca' but with 'f_pca'
    replaced by aligned version and gamI added
  """
  
  # Load packages
  import fdasrsf as fs
  import fdasrsf.utility_functions as uf
  import numpy as np
  
  # Extract necessary values
  time = aligned_train.time
  gam = aligned_train.gam
  pc_dirs = jfpca_train.f_pca
  
  # Determine necessary dimensions
  ndirs = pc_dirs.shape[1]
  npcs = pc_dirs.shape[2]
  
  # Compute gamma inverse
  gamI = uf.SqrtMeanInverse(gam)
  
  # Create empty array to store aligned principal directions
  aligned_pcdirs = np.zeros(pc_dirs.shape)
  
  # Align principal directions
  for pc in range(0, npcs):
    for pcdir in range(0, ndirs):
      aligned_pcdirs[:,pcdir,pc] = fs.warp_f_gamma(time=time,f=pc_dirs[:,pcdir,pc],gam=gamI)
  
  # Change PC directions to aligned version and add gamI to jfpca object
  jfpca_train.f_pca = aligned_pcdirs
  jfpca_train.gamI = gamI
  
  # Return the updated jfPCA results
  return(jfpca_train)  

def center_warping_funs(aligned_train):
  
  """ Function for centering warping functions obtained from 'time_warping'
  function from 'fdasrsf'
  
  Args:
    aligned_train: object output from 'time_warping'
  
  Returns:
      Object with the same structure as aligned_train but with qn, fn, and gam
      replaced by centered versions
  
  """
  
  # Load packages
  import fdasrsf as fs
  import fdasrsf.utility_functions as uf
  import numpy as np
  
  # Extract necessary values
  time = aligned_train.time
  gam = aligned_train.gam
  M = gam.shape[0]
  N = gam.shape[1]
  
  # Perform centering
  gamI = uf.SqrtMeanInverse(gam)
  gamI_dev = np.gradient(gamI, 1 / float(M - 1))
  time0 = (time[-1] - time[0]) * gamI + time[0]  
  mq = np.interp(time0, time, aligned_train.mqn) * np.sqrt(gamI_dev)
  for k in range(0, N):
      aligned_train.qn[:, k] = np.interp(time0, time, aligned_train.qn[:, k]) * np.sqrt(gamI_dev)
      aligned_train.fn[:, k] = np.interp(time0, time, aligned_train.fn[:, k])
      aligned_train.gam[:, k] = np.interp(time0, time, aligned_train.gam[:, k])
  
  # Return the centered results
  return(aligned_train)
