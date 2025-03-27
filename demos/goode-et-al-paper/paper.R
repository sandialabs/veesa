## ----setup, include = FALSE----------------------------------------------------------------------------------------------

# R markdown options
knitr::opts_chunk$set(
   echo = FALSE,
   message = FALSE,
   fig.align = "center",
   dpi = 600,
   warning = FALSE,
   error = FALSE
)

# More options for R markdown
knitr::opts_knit$set(eval.after = "fig.cap")

# Load packages
library(cowplot)
library(dplyr)
library(fdasrvf)
library(forcats)
library(ggplot2)
library(kableExtra)
library(latex2exp)
library(purrr)
library(randomForest)
library(reticulate)
library(stringr)
library(tidyr)
library(veesa)
library(wesanderson)

# Specify the conda environment to use - create environment using veesa_env.yml
use_condaenv(condaenv = "veesa", required = T)

# Specify colors for groups
col_2groups = wes_palettes$Royal1[2:1]
col_3groups = wes_palettes$Royal1[c(1,3,2)]
col_4groups = wes_palette(name = "Royal1", n = 20, type = "continuous")[c(1,2,4,7)]
col_5groups = wes_palette(name = "Royal1", n = 5, type = "continuous")[c(1,2,3,5,4)]

# Specify some high/low colors for contrast throughout
col_low = wes_palette(name = "Royal1", n = 30, type = "continuous")[4]
col_high = wes_palette(name = "Royal1", n = 30, type = "continuous")[29]

# Specify colors for PC direction plots
col_plus1 = "#784D8C"
col_plus2 = "#A289AE"
col_plus3 = "#CAC0D2"
col_minus1 = "#EA9B44"
col_minus2 = "#EBBC88"
col_minus3 = "#F3DABC"
col_pcdir_1sd = c(col_plus1, "black", col_minus1)
col_pcdir_2sd = c(col_plus2, col_plus1, "black", col_minus1, col_minus2)
col_pcdir_3sd = c(col_plus3, col_plus2, col_plus1, "black", col_minus1, col_minus2, col_minus3)

# Specify inkjet colors
col_inkjet = c("#0EB4AD", "#8f1883", "#d9c004")

# Specify the (base) font size (ch2_fs) and font (ff) for all plots
fs = 8
ff = "Helvetica"

# Specify a file path
fp = "~/OneDrive - Sandia National Laboratories/Documents/projects/veesa/"  #kjgoode
#fp = "~/OneDrive - Sandia National Laboratories/veesa/"  #jdtuck
#fp = "~/veesa/"



## # Load python packages
## import fdasrsf as fs
## import numpy as np
## import os
## import pandas as pd
## import pickle
## import sys
## 
## # Load functions from python packages
## from functools import reduce
## from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
## from sklearn.inspection import permutation_importance
## from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
## from sklearn.model_selection import train_test_split
## from sklearn.neural_network import MLPClassifier, MLPRegressor
## 
## # Load local code
## sys.path.insert(1, '../code/')
## import hct_functions as hct

## ----shifted-peaks-data--------------------------------------------------------------------------------------------------

# Extract times
shifted_peaks_times = unique(shifted_peaks$data$t)

# Determine relevant sample sizes
shifted_peaks_n = length(unique(shifted_peaks$data$id))
shifted_peaks_n_per_data = shifted_peaks$data |> distinct(id, data) |> count(data)
shifted_peaks_ntrain <-
  shifted_peaks_n_per_data |> filter(data == "Training") |> pull(n)
shifted_peaks_ntest <-
  shifted_peaks_n_per_data |> filter(data == "Testing") |> pull(n)

# Create long versions of training/testing data sets
shifted_peaks_train_long <-
  shifted_peaks$data |> filter(data == "Training")
shifted_peaks_test_long <-
  shifted_peaks$data |> filter(data == "Testing")

# Create wide versions of training data set
shifted_peaks_train_wide <-
  shifted_peaks_train_long |>
  select(-t) |>
  mutate(index = paste0("t", index)) |>
  pivot_wider(names_from = index, values_from = y)

# Create wide versions of testing data set
shifted_peaks_test_wide <-
  shifted_peaks_test_long |>
  select(-t) |>
  mutate(index = paste0("t", index)) |>
  pivot_wider(names_from = index, values_from = y)

# Convert training data to matrix (N x M) of M functions with N samples
shifted_peaks_train_matrix <-
  shifted_peaks_train_wide |>
  select(-data, -id, -group) |>
  as.matrix() |>
  t()

# Convert testing data to matrix (N x M) of M functions with N samples
shifted_peaks_test_matrix <-
  shifted_peaks_test_wide |>
  select(-data,-id,-group) |>
  as.matrix() |>
  t()


## ----shifted-peaks-align-jfpca-------------------------------------------------------------------------------------------
# Align the training data functions and apply jfPCA
shifted_peaks_train_esa_fp = paste0(fp, "data/shifted-peaks/shifted-peaks-train-esa.rds")
if (file.exists(shifted_peaks_train_esa_fp)) {
  shifted_peaks_train_esa = readRDS(shifted_peaks_train_esa_fp)
} else {
  shifted_peaks_train_esa <- veesa::prep_training_data(
    f = shifted_peaks_train_matrix,
    time = shifted_peaks_times,
    omethod = "DPo",
    fpca_method = "jfpca"
  )
  saveRDS(shifted_peaks_train_esa, shifted_peaks_train_esa_fp)
}

# Align the testing data functions and apply jfPCA based
# on training data application
shifted_peaks_test_esa_fp = paste0(fp, "data/shifted-peaks/shifted-peaks-test-esa.rds")
if (file.exists(shifted_peaks_test_esa_fp)) {
  shifted_peaks_test_esa = readRDS(shifted_peaks_test_esa_fp)
} else {
  shifted_peaks_test_esa <- veesa::prep_testing_data(
    f = shifted_peaks_test_matrix,
    time = shifted_peaks_times,
    train_prep = shifted_peaks_train_esa,
    omethod = "DPo"
  )
  saveRDS(shifted_peaks_test_esa, shifted_peaks_test_esa_fp)
}

# Center warping and aligned functions
shifted_peaks_centered_fp = paste0(fp, "data/shifted-peaks/shifted-peaks-jfpca-aligned.rds")
if (file.exists(shifted_peaks_centered_fp)) {
  shifted_peaks_centered = readRDS(shifted_peaks_centered_fp)
} else {
  shifted_peaks_centered = center_warping_funs(train_obj = shifted_peaks_train_esa)
  saveRDS(shifted_peaks_centered, shifted_peaks_centered_fp)
}

# Obtain aligned PC directions
shifted_peaks_jfpca_aligned_fp = paste0(fp, "data/shifted-peaks/shifted-peaks-jfpca-aligned.rds")
if (file.exists(shifted_peaks_jfpca_aligned_fp)) {
  shifted_peaks_jfpca_aligned = readRDS(shifted_peaks_jfpca_aligned_fp)
} else {
  shifted_peaks_jfpca_aligned = align_pcdirs(train_obj = shifted_peaks_train_esa)
  saveRDS(shifted_peaks_jfpca_aligned, shifted_peaks_jfpca_aligned_fp)
}


## ----fig1, fig.width = 23, fig.height = 10, out.width = '5.5in', fig.cap = "\\emph{Shifted Peaks Training Data.} The simulated training data functions (top left), true, cross-sectional, and aligned functional means (top right), aligned functions (bottom left), and warping functions (bottom right)."----

# Specify the figure font size
f1_fs = 28

# Specify an aspect ratio for all plots relating to the sim data
shifted_peaks_ar = 0.5

# Extract the aligned functions
colnames(shifted_peaks_centered$alignment$fn) = unique(shifted_peaks_train_long$id)
shifted_peaks_aligned_funs <-
  data.frame(shifted_peaks_centered$alignment$fn) |>
  mutate(t = shifted_peaks_times) |>
  pivot_longer(cols = -t, names_to = "id", values_to = "y_aligned") |>
  mutate(id = as.integer(str_remove(id, "X"))) |>
  right_join(shifted_peaks_train_long |> select(id, group) |> distinct() |> mutate(group = as.character(group)), by = "id") |>
  select(id, group, t, y_aligned)

# Compute cross-sectional mean of the observed functions
shifted_peaks_cs_means <-
  shifted_peaks_train_long |>
  group_by(t, group) |>
  summarise(mean_cs = mean(y), .groups = "drop") |>
  mutate(t = round(t,2)) |>
  arrange(group)

# Compute cross-sectional mean of the aligned functions
shifted_peaks_aligned_means <-
  shifted_peaks_aligned_funs |>
  group_by(t, group) |>
  summarise(
    mean_aligned = mean(y_aligned),
    .groups = "drop"
  ) |>
  mutate(group = factor(group, levels = c("1", "2")), t = round(t,2)) |>
  arrange(group)

# Plot the data
plot_shifted_peaks <-
  ggplot(shifted_peaks_train_long, aes(x = t, y = y, color = group, group = id)) +
  geom_line(alpha = 0.35) +
  labs(
    x = TeX('$t$'),
    y = TeX('$y$')
  ) +
  scale_color_manual(values = col_2groups) +
  theme_bw(base_family = ff, base_size = f1_fs) +
  theme(
    legend.position = "none",
    aspect.ratio = shifted_peaks_ar,
    axis.title = element_text(size = f1_fs),
    axis.text = element_text(size = f1_fs),
    title = element_text(size = f1_fs)
  ) +
  ylim(0, 1.4)

# Plot the means
plot_shifted_peaks_means <-
  full_join(shifted_peaks$true_means, shifted_peaks_cs_means, by = c("group", "t")) |>
  full_join(shifted_peaks_aligned_means, by = c("group", "t")) |>
  pivot_longer(names_to = "mean", cols = mean_true:mean_aligned) |>
  mutate(mean = fct_recode(factor(mean), "True" = "mean_true", "Cross-Sectional" = "mean_cs", "Aligned" = "mean_aligned")) |>
  mutate(mean = fct_relevel(mean, "True", "Cross-Sectional", "Aligned")) |>
  ggplot(aes(x = t, y = value, group = factor(mean):factor(group), color = group, linetype = mean)) +
  geom_line(aes(size = mean)) +
  labs(
    y = "y",
    x = TeX('$t$'),
    color = "Group",
    fill = "Group",
    linetype = "Mean Type",
    size = "Mean Type"
  ) +
  scale_color_manual(values = col_2groups) +
  scale_fill_manual(values = col_2groups) +
  scale_size_manual(values = c(1.5, 1.5, 1.5)) +
  scale_linetype_manual(values = c("solid", "dotted", "longdash")) +
  theme_bw(base_family = ff, base_size = f1_fs) +
  theme(
    aspect.ratio = shifted_peaks_ar,
    axis.title = element_text(size = f1_fs),
    axis.text = element_text(size = f1_fs),
    title = element_text(size = f1_fs),
    legend.text = element_text(size = f1_fs*1.1),
    legend.title = element_text(size = f1_fs*1.1),
    legend.key.width = unit(2.5,"cm")
  ) +
  ylim(0, 1.4) +
  guides(
    color = guide_legend(override.aes = list(linewidth = 1.75, alpha = 1)),
    linetype = guide_legend(override.aes = list(linewidth = 1.75, alpha = 1))
  )

# Plot the aligned functions
plot_shifted_peaks_aligned <-
  shifted_peaks_aligned_funs |>
  ggplot(aes(x = t, y = y_aligned, group = id, color = group)) +
  geom_line(alpha = 0.35) +
  labs(
    x = TeX('$t$'),
    y = "y"
  ) +
  scale_color_manual(values = col_2groups) +
  theme_bw(base_family = ff, base_size = f1_fs) +
  theme(
    legend.position = "none",
    aspect.ratio = shifted_peaks_ar,
    axis.title = element_text(size = f1_fs),
    axis.text = element_text(size = f1_fs),
    title = element_text(size = f1_fs)
  ) +
  ylim(0, 1.4)

# Plot the warping functions
plot_shifted_peaks_warping_funs <-
  shifted_peaks_centered$alignment$gam |>
  t() |>
  data.frame() |>
  mutate(id = 1:n()) |>
  pivot_longer(cols = -id, names_to = "index", values_to = "w") |>
  bind_cols(shifted_peaks_train_long |> select(group, t)) |>
  ggplot(aes(x = t, y = w, group = id, color = group)) +
  geom_line(alpha = 0.35) +
  scale_color_manual(values = col_2groups) +
  theme_bw(base_family = ff, base_size = f1_fs) +
  theme(
    legend.position = "none",
    aspect.ratio = shifted_peaks_ar,
    axis.title = element_text(size = f1_fs),
    axis.text = element_text(size = f1_fs),
    title = element_text(size = f1_fs)
  ) +
  labs(
    x = TeX("$t$"),
    y = "Warping function"
  )

# Extract the legend
legend <-
  get_plot_component(
    plot_shifted_peaks_means,
    'guide-box-right',
    return_all = TRUE
  )
plot_shifted_peaks_means <- plot_shifted_peaks_means + theme(legend.position = "none")

# Join the plots
plot_grid(
  plot_grid(
    plot_shifted_peaks,
    plot_shifted_peaks_means,
    plot_shifted_peaks_aligned,
    plot_shifted_peaks_warping_funs,
    ncol = 2,
    label_fontfamily = ff,
    label_size = f1_fs
  ),
  legend,
  ncol = 2,
  rel_widths = c(0.825,0.175)
)


## ----fig2, fig.width = 9.5, fig.height = 2.5, out.width = '5.75in', fig.cap = "\\emph{Shifted Peaks Data Principal Directions.} Principal direction plots for understanding the functional variability captured by jfPCs 1 and 2 from the shifted peaks data."----

# Specify the figure font size
f2_fs = 14

# Create figure
plot_pc_directions(
  fpc = 1:2,
  fdasrvf = shifted_peaks_jfpca_aligned$fpca_res,
  fpca_method = "jfpca",
  times = -shifted_peaks_times,
  linesizes = rep(0.75,5),
  alpha = 0.9
) +
  theme_bw(base_family = ff, base_size = f2_fs) +
  theme(
    strip.background = element_rect(color = "white", fill = "white"),
    legend.key.width = unit(1,"cm"),
    aspect.ratio = shifted_peaks_ar,
    axis.title = element_text(size = f2_fs),
    axis.text = element_text(size = f2_fs)
  ) +
  scale_color_manual(values = col_pcdir_2sd) +
  labs(x = TeX('$t$'), y = TeX('$y_g(t)$')) +
  guides(
    color = guide_legend(reverse = TRUE),
    linetype = guide_legend(reverse = TRUE),
    size = guide_legend(reverse = TRUE)
  )


## ----shifted-peaks-model-------------------------------------------------------------------------------------------------
# Prepare data for model
shifted_peaks_train_model_data <-
  shifted_peaks_train_esa$fpca_res$coef |>
  data.frame() |>
  mutate(group = factor(shifted_peaks_train_wide$group))
shifted_peaks_test_model_data <-
  t(shifted_peaks_test_esa$coef) |>
  data.frame() |>
  mutate(group = factor(shifted_peaks_test_wide$group))

# Random forest on jfPCA with aligned data
shifted_peaks_rf_jfpca_fp = paste0(fp, "results/shifted-peaks/shifted-peaks-rf-jfpca.rds")
if (file.exists(shifted_peaks_rf_jfpca_fp)) {
  shifted_peaks_rf_jfpca = readRDS(shifted_peaks_rf_jfpca_fp)
} else {
  set.seed(20210301)
  shifted_peaks_rf_jfpca <-
    randomForest::randomForest(
      formula = group ~ .,
      data = shifted_peaks_train_model_data,
      importance = TRUE)
  saveRDS(shifted_peaks_rf_jfpca, shifted_peaks_rf_jfpca_fp)
}

# Compute the model predictions on the test data
shifted_peaks_train_pred_jfpca = predict(shifted_peaks_rf_jfpca, shifted_peaks_train_model_data |> select(-group))
shifted_peaks_test_pred_jfpca = predict(shifted_peaks_rf_jfpca, shifted_peaks_test_model_data |> select(-group))

# Compute the accuracy on the training data
shifted_peaks_train_acc_jfpca = sum(shifted_peaks_train_pred_jfpca == shifted_peaks_train_model_data$group) / shifted_peaks_ntrain
shifted_peaks_test_acc_jfpca = sum(shifted_peaks_test_pred_jfpca == shifted_peaks_test_model_data$group) / shifted_peaks_ntest


## ----shifted-peaks-pfi---------------------------------------------------------------------------------------------------
# Specify the number of replications used to compute PFI with the example data
shifted_peaks_K = 10

# Compute PFI using accuracy with the VEESA pipeline (training data)
shifted_peaks_pfi_acc_jfpca_fp = paste0(fp, "results/shifted-peaks/shifted-peaks-pfi-acc-jfpca.rds")
if (file.exists(shifted_peaks_pfi_acc_jfpca_fp)) {
  shifted_peaks_pfi_acc_jfpca = readRDS(shifted_peaks_pfi_acc_jfpca_fp)
} else {
  set.seed(20210921)
  shifted_peaks_pfi_acc_jfpca <-
    compute_pfi(
      x = shifted_peaks_train_model_data |> select(-group),
      y = shifted_peaks_train_model_data$group,
      f = shifted_peaks_rf_jfpca,
      K = shifted_peaks_K,
      metric = "accuracy"
    )
  saveRDS(shifted_peaks_pfi_acc_jfpca, shifted_peaks_pfi_acc_jfpca_fp)
}

# Compute PFI using accuracy with the VEESA pipeline (testing data)
shifted_peaks_pfi_acc_jfpca_test_fp = paste0(fp, "results/shifted-peaks/shifted-peaks-pfi-acc-jfpca-test.rds")
if (file.exists(shifted_peaks_pfi_acc_jfpca_test_fp)) {
  shifted_peaks_pfi_acc_jfpca_test = readRDS(shifted_peaks_pfi_acc_jfpca_test_fp)
} else {
  set.seed(20210921)
  shifted_peaks_pfi_acc_jfpca_test <-
    compute_pfi(
      x = shifted_peaks_test_model_data |> select(-group),
      y = shifted_peaks_test_model_data$group,
      f = shifted_peaks_rf_jfpca,
      K = shifted_peaks_K,
      metric = "accuracy"
    )
  saveRDS(shifted_peaks_pfi_acc_jfpca_test, shifted_peaks_pfi_acc_jfpca_test_fp)
}

# Put pfi and proportion of variation in a data frame
shifted_peaks_prop_var_pfi <-
  data.frame(
    fpc = 1:length(shifted_peaks_times),
    prop_var = (shifted_peaks_train_esa$fpca_res$latent) ^ 2 /
      sum((shifted_peaks_train_esa$fpca_res$latent) ^ 2),
    pfi_train = shifted_peaks_pfi_acc_jfpca$pfi,
    pfi_test = shifted_peaks_pfi_acc_jfpca_test$pfi
  )

# Extract the jfPC with the highest PFI
shifted_peaks_train_top_fpc = shifted_peaks_prop_var_pfi |> filter(pfi_train == max(pfi_train)) |> pull(fpc)
shifted_peaks_train_top_pfi = shifted_peaks_prop_var_pfi |> filter(pfi_train == max(pfi_train)) |> pull(pfi_train)
shifted_peaks_test_top_fpc = shifted_peaks_prop_var_pfi |> filter(pfi_test == max(pfi_test)) |> pull(fpc)
shifted_peaks_test_top_pfi = shifted_peaks_prop_var_pfi |> filter(pfi_test == max(pfi_test)) |> pull(pfi_test)

# Put PFI reps in a data frame
shifted_peaks_prop_var_pfi_reps <-
  data.frame(
    fpc = 1:length(shifted_peaks_times),
    pfi_train = t(shifted_peaks_pfi_acc_jfpca$pfi_single_reps),
    pfi_test = t(shifted_peaks_pfi_acc_jfpca_test$pfi_single_reps)
  )


## ----fig3, fig.width = 8, fig.height = 5, out.width = '4.75in', fig.cap = "\\emph{Shifted Peaks Data jfPC Metrics.} (Top) Proportion of variation explained by jfPCs. (Middle and Bottom) Blue diamonds represent PFI values from the training and testing data, respectively. Boxplots represent the variability across repitions."----

# Specify the figure font size
f3_fs = 13

shifted_peaks_prop_var_pfi_single_long <-
  shifted_peaks_prop_var_pfi |>
  pivot_longer(cols = -fpc, names_to = "variable", values_to = "single") |>
  separate(variable, into = c("variable", "number"), sep = "\\.") |>
  select(-number)

shifted_peaks_prop_var_pfi_reps_long <-
  shifted_peaks_prop_var_pfi_reps |>
  pivot_longer(cols = -fpc, names_to = "variable", values_to = "reps") |>
  separate(variable, into = c("variable", "number"), sep = "\\.") |>
  select(-number)

shifted_peaks_prop_var_pfi_long  <-
  full_join(
    shifted_peaks_prop_var_pfi_single_long,
    shifted_peaks_prop_var_pfi_reps_long,
    by = join_by(fpc, variable)
  ) |>
  mutate(
    variable = fct_relevel(variable, "prop_var", "pfi_train", "pfi_test"),
    variable = fct_recode(
      variable,
      "Proportion of \nvariation" = "prop_var",
      "PFI (training)" = "pfi_train",
      "PFI (testing)" = "pfi_test"
    )
  )

ggplot(shifted_peaks_prop_var_pfi_long |> filter(fpc < 50)) +
  geom_boxplot(
    aes(x = fpc, y = reps, group = fpc)
  ) +
  geom_point(
    aes(x = fpc, y = single),
    size = 2.5,
    shape = 18,
    color = "steelblue"
  ) +
  facet_grid(variable ~ ., scales = "free_y", switch = "y") +
  labs(x = "jfPC") +
  theme_bw(base_family = ff, base_size = f3_fs) +
  theme(
    strip.placement = "outside",
    strip.background = element_rect(color = "white", fill = "white"),
    axis.title.y = element_blank(),
    axis.title.x = element_text(size = f3_fs),
    axis.text = element_text(size = f3_fs),
    title = element_text(size = f3_fs),
    strip.text = element_text(size = f3_fs)
  )




## ----hct-data------------------------------------------------------------------------------------------------------------
# Train/Test data dimensions
hct_train_dims = py_load_object(paste0(fp, "data/hct/hct-train-dims.pkl"))
hct_test_dims = py_load_object(paste0(fp, "data/hct/hct-test-dims.pkl"))
hct_total_dims = list()
hct_total_dims$full = hct_train_dims$train + hct_test_dims$test
hct_total_dims$h2o = hct_train_dims$h2o + hct_test_dims$h2o
hct_total_dims$exp = hct_train_dims$exp + hct_test_dims$exp
hct_total_dims$hp100 = hct_train_dims$hp100 + hct_test_dims$hp100
hct_total_dims$hp50 = hct_train_dims$hp50 + hct_test_dims$hp50
hct_total_dims$hp10 = hct_train_dims$hp10 + hct_test_dims$hp10

# Load subset of training data
hct_train_sub = py_load_object(paste0(fp, "data/hct/hct-sub-train.pkl"))


## 
## ## -----------------------------------------------------------------------------
## ##
## ## THE ANALYSIS FOR THE H-CT DATA WAS TOO COMPUTATIONALLY INTENSIVE TO RUN IN
## ## R MARKDOWN. INSTEAD, IT WAS RUN USING PYTHON CODE ON A LARGER COMPUTER. THE
## ## PYTHON SCRIPTS USED FOR THIS ANALYSIS CAN BE FOUND IN THE FOLDER ./code. THE
## ## FILES CONTAIN -hct- IN THE NAME, AND THE NUMBER AT THE BEGINNING OF THE FILE
## ## NAME INDICATE THE ORDER IN WHICH THE FILES SHOULD BE RUN.
## ##
## ## -----------------------------------------------------------------------------
## 

## ----hct-load-res--------------------------------------------------------------------------------------------------------
# Aligned/warped training data
hct_train_align_sub = py_load_object(paste0(fp, "data/hct/hct-sub-train-aligned-centered-sparam15.pkl"))
hct_train_warp_fun_sub = py_load_object(paste0(fp, "data/hct/hct-sub-train-warping-centered-sparam15.pkl"))

# Principal components from training data
hct_train_jfpca_pcs = py_load_object(paste0(fp, "data/hct/hct-train-jfpca-centered-pc-dirs-sparam15.pkl"))
hct_train_vfpca_pcs = py_load_object(paste0(fp, "data/hct/hct-train-vfpca-pc-dirs-sparam15.pkl"))

# Latent values from training data
hct_train_jfpca_latent = py_load_object(paste0(fp, "data/hct/hct-train-jfpca-latent-sparam15.pkl"))
hct_train_vfpca_latent = py_load_object(paste0(fp, "data/hct/hct-train-vfpca-latent-sparam15.pkl"))

# Model metrics
hct_jfpca_pred_metrics_train = py_load_object(paste0(fp, "results/hct/hct-train-pred-and-metrics-jfpca-sparam15.pkl"))
hct_jfpca_pred_metrics_test = py_load_object(paste0(fp, "results/hct/hct-test-pred-and-metrics-jfpca-sparam15.pkl"))
hct_vfpca_pred_metrics_train = py_load_object(paste0(fp, "results/hct/hct-train-pred-and-metrics-vfpca-sparam15.pkl"))
hct_vfpca_pred_metrics_test = py_load_object(paste0(fp, "results/hct/hct-test-pred-and-metrics-vfpca-sparam15.pkl"))

# PFI
hct_jfpca_nn_pfi = py_load_object(paste0(fp, "results/hct/hct-test-pfi-jfpca-sparam15.pkl"))
hct_vfpca_nn_pfi = py_load_object(paste0(fp, "results/hct/hct-test-pfi-vfpca-sparam15.pkl"))


## ----hct-prep-data-------------------------------------------------------------------------------------------------------
# Create long version of aligned data subset
hct_train_align_sub_long <-
  hct_train_align_sub |>
  pivot_longer(cols = -id, names_to = "frequency", values_to = "aligned") |>
  mutate(frequency = as.integer(frequency))

# Create long version of warping function subset
hct_train_warp_fun_sub_long <-
  hct_train_warp_fun_sub |>
  pivot_longer(cols = -id, names_to = "frequency", values_to = "warping") |>
  mutate(frequency = as.integer(frequency))

# Join aligned and warping functions with training data
hct_train_sub$frequency = unlist(hct_train_sub$frequency)
hct_train_sub_full <-
  hct_train_sub |>
  left_join(hct_train_align_sub_long, by = c("id", "frequency")) |>
  left_join(hct_train_warp_fun_sub_long, by = c("id", "frequency")) |>
  mutate(material = fct_relevel(material, "explosive", "h2o", "hp10", "hp50", "hp100"))


## ----hct-calculations----------------------------------------------------------------------------------------------------
# Compute the number of observations per material subset
n_obs_sub <-
  hct_train_sub |>
  select(material, id) |>
  distinct() |>
  count(material)

# Print a message if the subsets have different sizes
if (sum(n_obs_sub$n[1] != n_obs_sub$n) != 0) {
  print("Error: H-CT subsets have different sizes")
}

# Compute cross-sectional means from aligned functions
hct_train_sub_means <-
  hct_train_sub_full |>
  group_by(material, frequency_norm) |>
  summarise(mean = mean(aligned), .groups = "drop") |>
  mutate(material = fct_relevel(factor(material), "explosive", "h2o", "hp10", "hp50", "hp100")) |>
  mutate(
    material = fct_recode(
      material,
      "Explosive" = "explosive",
      "H2O" = "h2o",
      "H2O2 (10%)" = "hp10",
      "H2O2 (50%)" = "hp50",
      "H2O2 (100%)" = "hp100"
    )
  )

# Function for computing warping functions mean using fdasrvf
get_warp_mean <- function(m, df) {

  # Subset for specified material and put in matrix
  warp_mat <-
    df |>
    filter(material == m) |>
    select(id, frequency, warping) |>
    pivot_wider(id_cols = id,
                names_from = frequency,
                values_from = warping) |>
    select(-id) |>
    as.matrix() |>
    t()

  # Apply sqrt mean function
  res = SqrtMean(gam = warp_mat)

  # Extract warping function mean and put in df
  warp_mean = data.frame(var = res$gam_mu)
  colnames(warp_mean) = m

  # Return warping mean df
  warp_mean

}

# Compute the warping functions means from the subset for each material
hct_warp_means_sub <-
  map_dfc(
    .x = c("h2o", "explosive", "hp100", "hp50", "hp10"),
    .f = get_warp_mean,
    df = hct_train_sub_full
  ) |>
  mutate(frequency_norm = sort(unique(hct_train_sub_full$frequency_norm))) |>
  pivot_longer(
    cols = -frequency_norm,
    names_to = "material",
    values_to = "warp_mean"
  ) |>
  mutate(material = fct_relevel(factor(material), "explosive", "h2o", "hp10", "hp50", "hp100")) |>
  mutate(
    material = fct_recode(
      material,
      "Explosive" = "explosive",
      "H2O" = "h2o",
      "H2O2 (10%)" = "hp10",
      "H2O2 (50%)" = "hp50",
      "H2O2 (100%)" = "hp100"
    )
  )

# Compute the proportion of variation
hct_prop_var_jfpca <-
  data.frame(latent = hct_train_jfpca_latent) |>
  mutate(index = 1:n(),
         prop_var = (latent^2) / (sum(latent^2)))
hct_prop_var_vfpca <-
  data.frame(latent = hct_train_vfpca_latent) |>
  mutate(index = 1:n(),
         prop_var = (latent^2) / (sum(latent^2)))

# Function for identifying a certain number of top jfPCs
get_top_vars <- function(pfi, nvars) {
  data.frame(pfi_mean = pfi$importances_mean) |>
    mutate(x = 1:n()) |>
    arrange(desc(pfi_mean)) |>
    slice(1:nvars) |>
    pull(x)
}

# Specify number of top fPCs to consider
hct_n_top_pcs = 6

# Identify the top fPCs
hct_top_jfpcs = get_top_vars(pfi = hct_jfpca_nn_pfi, nvars = hct_n_top_pcs)
hct_top_vfpcs = get_top_vars(pfi = hct_vfpca_nn_pfi, nvars = hct_n_top_pcs)



## ----fig4, fig.height = 7, fig.width = 14, out.width = "5.5in", fig.cap = "\\emph{H-CT Data Material Signatures.} Observed (top row), smoothed and aligned (middle row), and warping functions (bottom row) of a subset of 1,000 H-CT signatures for each material."----

# Specify the figure font size
f5_fs = 19

# Plot subset of signatures
hct_train_sub_full |>
  rename("Observed \nFunctions" = "value",
         "Smoothed and \nAligned Functions" = "aligned",
         "Warping \nFunctions" = "warping") |>
  pivot_longer(
    cols = c("Observed \nFunctions", "Smoothed and \nAligned Functions", "Warping \nFunctions"),
    names_to = "situation"
  ) |>
  mutate(situation = fct_relevel(
    situation,
    "Observed \nFunctions",
    "Smoothed and \nAligned Functions",
    "Warping \nFunctions"
  )) |>
  mutate(
    material = fct_recode(
      material,
      "Explosive" = "explosive",
      "H2O" = "h2o",
      "H2O2 (10%)" = "hp10",
      "H2O2 (50%)" = "hp50",
      "H2O2 (100%)" = "hp100"
    )
  ) |>
  ggplot(aes(
    x = frequency_norm,
    y = value,
    group = id,
    color = material
  )) +
  geom_line(alpha = 0.1, size = 0.1) +
  facet_grid(situation ~ material, scales = "free_y") +
  theme_bw(base_family = ff, base_size = f5_fs) +
  theme(
    legend.key.width = unit(1, "cm"),
    strip.background = element_rect(color = "white", fill = "white"),
    aspect.ratio = 0.75,
    legend.position = "none",
    axis.title = element_text(size = f5_fs),
    axis.text = element_text(size = f5_fs*0.7),
    title = element_text(size = f5_fs),
    strip.text = element_text(size = f5_fs*0.8)
  ) +
  scale_color_manual(values = col_5groups) +
  guides(color = guide_legend(override.aes = list(alpha = 1, size = 0.5))) +
  labs(color = "Material", y = "Intensity", x = "Normalized Frequency")



## ----fig5, fig.width = 17, fig.height = 6, out.width = "4.75in", fig.cap = "\\emph{H-CT Data Material Means.} For each material, cross-sectional functional means of the aligned functions (left) and Karcher means of the warping functions (right)."----

# Specify the figure font size
f6_fs = 22

plot_hct_train_means <-
  hct_train_sub_means |>
  ggplot(aes(x = frequency_norm, y = mean, color = material)) +
  geom_line(size = 1.25) +
  theme_bw(base_family = ff, base_size = f6_fs) +
  scale_color_manual(values = col_5groups) +
  theme(
    aspect.ratio = 0.45,
    legend.position = "none",
    axis.title.y = element_blank(),
    axis.title = element_text(size = f6_fs),
    axis.text = element_text(size = f6_fs),
    title = element_text(size = f6_fs * 0.9),
    strip.text = element_text(size = f6_fs)
  ) +
  labs(
    x = "Normalized Frequency",
    title = "Aligned Cross-Sectional Means",
    color = "Material"
  )

plot_hct_train_warp_means <-
  hct_warp_means_sub |>
  ggplot(aes(x = frequency_norm, y = warp_mean, color = material)) +
  geom_line(size = 1.25) +
  theme_bw(base_family = ff, base_size = f6_fs) +
  theme(
    aspect.ratio = 1,
    axis.title.y = element_blank(),
    axis.title.x = element_text(size = f6_fs),
    axis.text = element_text(size = f6_fs),
    title = element_text(size = f6_fs * 0.9),
    strip.text = element_text(size = f6_fs),
    legend.text = element_text(size = f6_fs),
    legend.title = element_text(size = f6_fs),
    legend.position = "bottom",
    legend.key.width = unit(2, 'cm')
  ) +
  scale_color_manual(values = col_5groups) +
  guides(color = guide_legend(override.aes = list(linewidth = 1.25, alpha = 1, size = 0.5))) +
  labs(
    x = "Normalized Frequency",
    title = "Warping Karcher Means",
    color = "Material"
  )

# Extract the legend
f6_legend <-
  get_plot_component(
    plot_hct_train_warp_means,
    'guide-box-bottom',
    return_all = TRUE
  )
plot_hct_train_warp_means <-
  plot_hct_train_warp_means +
  theme(legend.position = "none")

# Join the plots
plot_grid(
  plot_grid(
    plot_hct_train_means,
    plot_hct_train_warp_means,
    rel_widths = c(0.6, 0.4),
    label_fontfamily = ff
  ),
  f6_legend,
  ncol = 1,
  rel_heights = c(0.95, 0.05)
)


## ----fig6, fig.width = 15, fig.height = 3.5, out.width = '5.75in', fig.cap = "\\emph{H-CT Data vfPC Metrics.} Proportion of variation and PFI values associated with vfPCs. jfPCs with the 6 highest PFI values are labeled and colored.", warning = FALSE----

# Specify the figure font size
f7_fs = 20

# Prepare proportion of variability and PFI for vfPCA for plotting 
hct_pv_pfi <-
  hct_prop_var_vfpca |>
  mutate(fpca = "Vertical fPCA", pfi = hct_vfpca_nn_pfi$importances_mean) |>
  select(-latent) |>
  rename(fpc = index) |>
  pivot_longer(cols = -c(fpc, fpca), names_to = "variable") |>
  mutate(
    variable = fct_relevel(variable, "prop_var", "pfi"),
    variable = fct_recode(
      variable,
      "Proportion of \nVariation" = "prop_var",
      "PFI" = "pfi"
    )
  )

# Prepare labels
hct_pfi_labels <-
  hct_pv_pfi |>
  filter(variable == "PFI") |>
  filter(fpc %in% hct_top_vfpcs) |>
  mutate(label_x = fpc, label_y = value, label_id = 1:n()) |>
  mutate(label_y = ifelse(label_id == 1, label_y + 0.03, label_y)) |>
  mutate(label_x = ifelse(label_id == 2, label_x - 3, label_x)) |>
  mutate(label_y = ifelse(label_id == 3, label_y + 0.03, label_y)) |>
  mutate(label_x = ifelse(label_id == 4, label_x + 2.5, label_x)) |>
  mutate(label_y = ifelse(label_id == 5, label_y + 0.02, label_y)) |>
  mutate(label_y = ifelse(label_id == 6, label_y + 0.03, label_y))

# Create plots of proportion of variation explained and PFI values
left_join(hct_pv_pfi, hct_pfi_labels) |>
  mutate(has_label = ifelse(is.na(label_x), F, T)) |>
  ggplot(aes(x = fpc, y = value)) +
  geom_segment(aes(xend = fpc, yend = 0, color = has_label), size = 0.75) +
  facet_wrap(. ~ variable, scales = "free_y", switch = "y") +
  geom_text(aes(x = label_x, y = label_y, label = fpc, color = has_label),
            size = f7_fs / 3) +
  labs(x = "Principal Component") +
  scale_color_manual(values = c("black", col_5groups[1])) +
  theme_bw(base_family = ff, base_size = f7_fs) +
  theme(
    strip.placement = "outside",
    strip.background =
      element_rect(color = "white", fill = "white"),
    axis.title.y = element_blank(),
    axis.title.x = element_text(size = f7_fs),
    axis.text = element_text(size = f7_fs),
    title = element_text(size = f7_fs),
    strip.text = element_text(size = f7_fs), 
    legend.position = "none"
  )


## ----fig7, fig.width = 28, fig.height = 14, out.width = "5.5in", fig.cap = "\\emph{H-CT Data Principal Directions.} Principal directions from the six vfPCs with the highest PFI from the H-CT data example."----

# Specify the figure font size
f8_fs = 40

plot_pc_directions(
  fpcs = hct_top_vfpcs,
  fdasrvf = list(
    "f_pca" = hct_train_vfpca_pcs[, 2:6, ],
    "latent" = hct_train_vfpca_latent
  ),
  times = unique(hct_train_sub$frequency_norm),
  fpca_method = "vfpca",
  alpha = 0.9,
  nrow = 2,
  linesizes = rep(1.75, 5),
  linetype = FALSE
) +
  theme_bw(base_size = f8_fs, base_family = ff) +
  theme(
    strip.background = element_rect(color = "white", fill = "white"),
    aspect.ratio = 0.45,
    legend.position = "bottom",
    axis.title = element_text(size = f8_fs),
    axis.text = element_text(size = f8_fs*0.8),
    title = element_text(size = f8_fs),
    strip.text = element_text(size = f8_fs),
    legend.text = element_text(size = f8_fs),
    legend.title = element_text(size = f8_fs),
    legend.key.width = unit(2, 'cm')
  ) +
  guides(color = guide_legend(override.aes = list(linewidth = 2))) +
  scale_color_manual(values = col_pcdir_2sd) +
  labs(
    x = "Normalized Frequency",
    y = "Intensity",
    color = "Material",
    size = "Material",
    linetype = "Material"
  )


## ----inkjet-data-cleaning------------------------------------------------------------------------------------------------

# File path for inkjet data
inkjet_fp = paste0(fp, "data/inkjet/inkjet-cleaned.csv")

# Create cleaned version of data (if not already created)
if (!file.exists(inkjet_fp)) {

  # Load the raw data
  inkjet_cyan_raw <- read.csv(paste0(fp, "data/inkjet_raw/RamanInkjet_PrelDataNoBsln1CYANrows.csv"))
  inkjet_magenta_raw <- read.csv(paste0(fp, "data/inkjet_raw/RamanInkjet_PrelDataNoBsln2MAGENTArows.csv"))
  inkjet_yellow_raw <- read.csv(paste0(fp, "data/inkjet_raw/RamanInkjet_PrelDataNoBsln3YELLOWrows.csv"))

  # Add printer labels to the yellow data
  inkjet_yellow_raw <-
    inkjet_yellow_raw |>
    mutate(
      code = case_when(
        str_detect(sample, "s003") ~ "P01",
        str_detect(sample, "s011") ~ "P02",
        str_detect(sample, "s012") ~ "P03",
        str_detect(sample, "s015") ~ "P04",
        str_detect(sample, "s001") ~ "P05",
        str_detect(sample, "s002") ~ "P06",
        str_detect(sample, "s016") ~ "P07",
        str_detect(sample, "s010") ~ "P08",
        str_detect(sample, "s014") ~ "P09",
        str_detect(sample, "s004") ~ "P10",
        str_detect(sample, "s005") ~ "P11"
      )
    ) |>
    select(sample, code, everything())

  # Join the three colors
  inkjet_colors_joined = bind_rows(inkjet_cyan_raw, inkjet_magenta_raw, inkjet_yellow_raw)

  # Clean up the data
  inkjet <-
    inkjet_colors_joined |>
    rename("id" = "sample", "printer" = "code") |>
    mutate(printer = as.numeric(str_remove(printer, "P"))) |>
    mutate(id_copy = id) |>
    separate(id_copy,
             sep = 4,
             into = c("ss_printer_code", "sample")) |>
    select(-ss_printer_code) |>
    separate(col = sample,
             into = c("color", "sample"),
             sep = 1) |>
    mutate(sample = as.numeric(sample)) |>
    select(id, printer, sample, color, everything()) |>
    pivot_longer(
      cols = c(-id, -printer, -color, -sample),
      names_to = "spectra",
      values_to = "intensity"
    ) |>
    mutate(spectra = str_remove(spectra, "X")) |>
    mutate(spectra = -as.numeric(spectra)) |>
    arrange(printer, color, sample)

  # Save cleaned data
  write.csv(inkjet, inkjet_fp, row.names = F)

}

# Load inkjet data
inkjet = read.csv(inkjet_fp)



## ----inkjet-cv-folds-----------------------------------------------------------------------------------------------------

# File path for cross validation folds
inkjet_folds_fp = paste0(fp, "data/inkjet/inkjet-cv-folds.csv")

# Create cross validation folds if not already created
if (!file.exists(inkjet_folds_fp)) {

  # Determine number of printers
  inkjet_n_printers = length(unique(inkjet$printer))

  # Specify number of replications of three-fold cross validation
  inkjet_n_reps = 10

  # Specify fold options for every 7 observations within a printer
  inkjet_fold_ids = c(1,1,1,2,2,3,3)

  # Randomly assign fold numbers
  set.seed(20221221)
  inkjet_folds <-
    map_df(
      .x = 1:inkjet_n_reps,
      .f = function(r) {
        map_df(
          .x = 1:inkjet_n_printers,
          .f = function(p) {
            inkjet |>
              distinct(printer, sample) |>
              filter(printer == p) |>
              mutate(rep = r) |>
              mutate(fold = sample(inkjet_fold_ids, length(inkjet_fold_ids), replace = F))
          }
        )
      }
    )

  # Save data with CV folds as CSV
  write.csv(inkjet_folds, inkjet_folds_fp, row.names = F)

}

# Load cross validation folds
inkjet_folds = read.csv(inkjet_folds_fp)



## 
## ## -----------------------------------------------------------------------------
## ##
## ## THE ESA PROCESSING OF THE INKJET DATA WAS TOO COMPUTATIONALLY INTENSIVE TO
## ## RUN IN R MARKDOWN. INSTEAD, IT WAS RUN USING PYTHON CODE ON A LARGER
## ## COMPUTER. THE PYTHON SCRIPTS USED FOR THIS CAN BE FOUND IN THE FOLDER ./code.
## ## THE FILES CONTAIN -inkjet- IN THE NAME, AND THE NUMBER AT THE BEGINNING OF
## ## THE FILE NAME INDICATES THE ORDER IN WHICH THE FILES SHOULD BE RUN.
## ##
## ## -----------------------------------------------------------------------------
## 

## ----inkjet-scenarios----------------------------------------------------------------------------------------------------
# File paths for saving CV scenarios
fp_inkjet_cv_srfc = paste0(fp, "results/inkjet/inkjet-cv-srfc.rds")
fp_inkjet_cv_rfc = paste0(fp, "results/inkjet/inkjet-cv-rfc.rds")

# Get cross validation test fold predictions
if (!file.exists(fp_inkjet_cv_srfc) | !file.exists(fp_inkjet_cv_rfc)) {

  # Specify smoothing values, reps, test fold numbers and color
  inkjet_s = c(0, 5, 10, 15, 20, 25, 30, 35)
  inkjet_reps = sort(unique(inkjet_folds$rep))
  inkjet_test_folds = sort(unique(inkjet_folds$fold))
  inkjet_colors = c("c", "m", "y")

  # Specify parameters for random forests
  inkjet_rf_params <-
    expand.grid(
      pcs = seq(10, 100, 10),
      ntrees = c(50, 100, 250, 500, 1000)
    )

  # Get all combinations of smoothing parameter, rep, fold, and color
  inkjet_smooth_rep_fold_color <-
    expand_grid(
      s = inkjet_s,
      rep = inkjet_reps,
      test_fold = inkjet_test_folds,
      color = inkjet_colors
    )

  # Convert to a list
  inkjet_smooth_rep_fold_color_list <-
    list(
      s = inkjet_smooth_rep_fold_color$s,
      rep = inkjet_smooth_rep_fold_color$rep,
      test_fold = inkjet_smooth_rep_fold_color$test_fold,
      color = inkjet_smooth_rep_fold_color$color
    )

  # Keep only values of rep, test fold, and colors
  inkjet_rep_fold_color <-
    inkjet_smooth_rep_fold_color |>
    distinct(rep, test_fold, color)

  # ... and convert to a list
  inkjet_rep_fold_color_list <-
    list(
      rep = inkjet_rep_fold_color$rep,
      test_fold = inkjet_rep_fold_color$test_fold,
      color = inkjet_rep_fold_color$color
    )

  # Save CV scenarios
  saveRDS(inkjet_smooth_rep_fold_color_list, fp_inkjet_cv_srfc)
  saveRDS(inkjet_rep_fold_color_list, fp_inkjet_cv_rfc)

}

# Load CV scenarios
inkjet_smooth_rep_fold_color_list = readRDS(fp_inkjet_cv_srfc)
inkjet_rep_fold_color_list = readRDS(fp_inkjet_cv_rfc)



## ----inkjet-cv-preds-----------------------------------------------------------------------------------------------------

# File path for saving CV predictions
fp_inkjet_cv_preds = paste0(fp, "results/inkjet/inkjet-cv-preds.rds")

# Get cross validation test fold predictions
if (!file.exists(fp_inkjet_cv_preds)) {

  # Function for loading inkjet CV objects
  load_inkjet <- function(rep, test_fold, color, data_type, fp, smoothing) {
    py_load_object(
      filename = paste0(
        fp,
        "data/inkjet/inkjet-cv-s",
        smoothing,
        "-rep_",
        rep,
        "-fold_",
        test_fold,
        "-color_",
        color,
        "-",
        data_type,
        ".pkl"
      )
    )
  }

  # Function for implementing cross validation for inkjet data
  get_inkjet_rf_cv_preds <- function(s, rep_curr, test_fold, color, pcs, ntrees) {

    # Load training data
    train <-
      load_inkjet(
        smoothing = s,
        rep = rep_curr,
        test_fold = test_fold,
        color = color,
        data_type = "jfpca-train",
        fp = fp
      )

    # Load testing data
    test <-
      load_inkjet(
        smoothing = s,
        rep = rep_curr,
        test_fold = test_fold,
        color = color,
        data_type = "aligned-jfpca-test",
        fp = fp
      )

    # Extract training data printer and sample values
    train_printer_sample <-
      inkjet_folds |>
      filter(rep == rep_curr, fold != test_fold)

    # Extract testing data printer and sample values
    test_printer_sample <-
      inkjet_folds |>
      filter(rep == rep_curr, fold == test_fold)

    # Train models and compute CV predictions
    set.seed(20230630)
    inkjet_cv_preds <-
      map2_df(
        .x = pcs,
        .y = ntrees,
        .f = fit_inkjet_rf_and_predict,
        train = train,
        train_printer_sample = train_printer_sample,
        test = test,
        test_printer_sample = test_printer_sample
      ) |>
      mutate(
        color = color,
        s = s,
        rep = rep_curr,
        test_fold = test_fold
      ) |>
      select(color, s, rep, test_fold, everything())

    # Return the results
    return(inkjet_cv_preds)

  }

  # Function for fitting random forests and getting test data predictions
  fit_inkjet_rf_and_predict <- function(pcs, ntrees, train, train_printer_sample, test, test_printer_sample) {

    # Create a dataframe with training data
    rf_data <- data.frame(
      printer = factor(train_printer_sample$printer),
      train$coef[,1:pcs]
    )

    # Train random forest
    rf = randomForest(printer ~ ., data = rf_data, ntree = ntrees)

    # Get test fold predictions
    preds = predict(rf, data.frame(test$coef[,1:pcs]))

    # Put predictions in a dataframe
    preds <-
      data.frame(
        pcs = pcs,
        ntrees = ntrees,
        printer = test_printer_sample$printer,
        sample = test_printer_sample$sample,
        pred = as.numeric(as.vector(preds))
      )

  }

  # Implement CV for all specified scenarios
  inkjet_cv_preds <-
    pmap_df(
      .l = inkjet_smooth_rep_fold_color_list,
      .f = get_inkjet_rf_cv_preds,
      pcs = inkjet_rf_params$pcs,
      ntrees = inkjet_rf_params$ntrees
    )

  # Save CV predictions
  saveRDS(inkjet_cv_preds, fp_inkjet_cv_preds)

}

# Load CV results
inkjet_cv_preds <-
  readRDS(fp_inkjet_cv_preds) |>
  mutate(
    color = fct_recode(color, "Cyan" = "c", "Magenta" = "m", "Yellow" = "y")
  )


## ----inkjet-cv-results---------------------------------------------------------------------------------------------------

# File for inkjet data cross validation accuracies
inkjet_cv_acc_fp = paste0(fp, "results/inkjet/inkjet-cv-accuracy.csv")
inkjet_cv_acc_summary_fp = paste0(fp, "results/inkjet/inkjet-cv-accuracy-summary.csv")
inkjet_res_best_fp = paste0(fp, "results/inkjet/inkjet-cv-res-best.csv")
inkjet_res_worst_fp = paste0(fp, "results/inkjet/inkjet-cv-res-worst.csv")

# Compute cross validation accuracies
if (!file.exists(inkjet_cv_acc_fp) | !file.exists(inkjet_cv_acc_summary_fp)) {

  # Compute cross validation accuracies (separately for each replicate)
  inkjet_cv_acc <-
    inkjet_cv_preds |>
    summarise(
      acc = sum(pred == printer) / length(printer),
      .by = c(color, s, pcs, ntrees, rep)
    )

  # Compute cross validation accuracies (summaries over replicates)
  inkjet_cv_acc_summary <-
    inkjet_cv_acc |>
    summarise(
      acc_ave = mean(acc),
      acc_sd = sd(acc),
      acc_min = min(acc),
      acc_max = max(acc),
      .by = c(s, color, pcs, ntrees)
    ) |>
    mutate(color = factor(color))

  # Determine top scenarios from CV results based on highest average accuracy
  inkjet_res_best <-
    inkjet_cv_acc_summary |>
    arrange(color, desc(acc_ave)) |>
    slice(1, .by = color) |>
    mutate_at(.vars = vars(acc_ave:acc_max), .funs = round, digits = 4)

  # Determine bottom scenarios from CV results based on lowest average accuracy
  inkjet_res_worst <-
    inkjet_cv_acc_summary |>
    arrange(color, acc_ave) |>
    slice(1, .by = color) |>
    mutate_at(.vars = vars(acc_ave:acc_max), .funs = round, digits = 4)

  # Save test fold accuracies (and best/worst scenarios)
  write.csv(inkjet_cv_acc, inkjet_cv_acc_fp, row.names = F)
  write.csv(inkjet_cv_acc_summary, inkjet_cv_acc_summary_fp, row.names = F)
  write.csv(inkjet_res_best, inkjet_res_best_fp, row.names = F)
  write.csv(inkjet_res_worst, inkjet_res_worst_fp, row.names = F)

}

# Load cross validation accuracies
inkjet_cv_acc = read.csv(inkjet_cv_acc_fp)
inkjet_cv_acc_summary = read.csv(inkjet_cv_acc_summary_fp)
inkjet_res_best = read.csv(inkjet_res_best_fp)
inkjet_res_worst = read.csv(inkjet_res_worst_fp)



## ----inkjet-pfi----------------------------------------------------------------------------------------------------------

# Files for storing inkjet PFI values
inkjet_pfi_best_fp = paste0(fp, "results/inkjet/inkjet-pfi-best.pkl")
inkjet_pfi_worst_fp = paste0(fp, "results/inkjet/inkjet-pfi-worst.pkl")

# Compute PFI for inkjet data
if (!file.exists(inkjet_pfi_best_fp) | !file.exists(inkjet_pfi_worst_fp)) {

  # Extract vector of inkjet printers
  inkjet_printer = inkjet |> distinct(printer, sample) |> pull(printer)

  # Function for computing PFI for a specific inkjet printer scenario
  get_inkjet_pfi <- function(s, color, pcs, ntrees) {
    set.seed(20230705)
    s = ifelse(s < 10, paste0(0, s), s)
    jfpca = py_load_object(paste0(fp, "data/inkjet/inkjet-s", s, "-jfpca-", tolower(color), ".pkl"))
    train = data.frame(printer = factor(inkjet_printer), jfpca$coef[,1:pcs])
    rf = randomForest(printer ~ ., data = train, ntree = ntrees)
    pfi <-
      veesa::compute_pfi(
        x = train |> select(-printer),
        y = train$printer,
        f = rf,
        K = 10,
        metric = "logloss"
      )
    pfi_df <-
      data.frame(
        color = color,
        pc = 1:pcs,
        pfi = pfi$pfi,
        pfi_reps = t(pfi$pfi_single_reps)
      )
    return(list(jfpca = jfpca, pfi_df = pfi_df))
  }

  # Compute PFI for the best inkjet printer scenarios
  inkjet_pfi_best <-
    pmap(
      .l = inkjet_res_best |> select(s, color, pcs, ntrees),
      .f = get_inkjet_pfi
    )

  # Compute PFI for the worst inkjet printer scenarios
  inkjet_pfi_worst <-
    pmap(
      .l = inkjet_res_worst |> select(s, color, pcs, ntrees),
      .f = get_inkjet_pfi
    )

  # Save inkjet PFI results
  py_save_object(inkjet_pfi_best, inkjet_pfi_best_fp)
  py_save_object(inkjet_pfi_worst, inkjet_pfi_worst_fp)

}

# Load inkjet PFI results
inkjet_pfi_best = py_load_object(inkjet_pfi_best_fp, convert = TRUE)
inkjet_pfi_worst = py_load_object(inkjet_pfi_worst_fp, convert = TRUE)


## ----inkjet-paper-results------------------------------------------------------------------------------------------------

# File for results from Buzzini paper
inkjet_paper_res_fp = paste0(fp, "results/inkjet/inkjet-paper-res.csv")
inkjet_paper_res_best_fp = paste0(fp, "results/inkjet/inkjet-paper-res-best.csv")

# Compute cross validation accuracies
if (!file.exists(inkjet_paper_res_fp)) {

  # Put results from paper in a data frame
  inkjet_paper_res <-
    data.frame(
      color = rep(c("Cyan", "Magenta", "Yellow"), each = 6),
      method = rep(rep(c("PCA + LDA", "PLSDA", "Sparse LDA"), each = 2), 3),
      baseline_corr = rep(c("No", "Yes"), 9),
      acc_ave = c(0.86, 0.87, 0.88, 0.87, 0.91, 0.89, 0.88, 0.87, 0.92, 0.91,
                  0.92, 0.92, 0.87, 0.87, 0.88, 0.87, 0.92, 0.91),
      acc_sd = c(0.08, 0.07, 0.06, 0.06, 0.05, 0.06, 0.07, 0.06, 0.05, 0.06,
                 0.06, 0.06, 0.05, 0.05, 0.06, 0.06, 0.04, 0.05),
      acc_min = c(0.65, 0.74, 0.73, 0.71, 0.81, 0.79, 0.64, 0.76, 0.81, 0.79,
                  0.76, 0.76, 0.76, 0.79, 0.76, 0.76, 0.88, 0.77),
      acc_max = c(1.00, 1.00, 0.96, 0.96, 1.00, 1.00, 1.00, 0.96, 1.00, 1.00,
                  1.00, 1.00, 1.00, 0.96, 1.00, 0.96, 1.00, 0.96)
    ) |>
    mutate(method_baseline = paste(method, "+", baseline_corr))

  # Determine ‘best’ results from paper methods based on highest accuracy
  # and highest minimum accuracy
  inkjet_paper_res_best <-
    inkjet_paper_res |>
    filter(acc_ave == max(acc_ave), .by = color) |>
    filter(acc_min == max(acc_min), .by = color)

  # Save results from Buzzini paper
  write.csv(inkjet_paper_res, inkjet_paper_res_fp, row.names = F)
  write.csv(inkjet_paper_res_best, inkjet_paper_res_best_fp, row.names = F)

}

# Load results from Buzzini paper
inkjet_paper_res = read.csv(inkjet_paper_res_fp)
inkjet_paper_res_best = read.csv(inkjet_paper_res_best_fp)



## ----fig8, fig.height = 12, fig.width = 35, out.width = "6.25in", fig.cap = "\\emph{Inkjet Data Signatures.} Raman spectra signatures from 11 inkjet printers for the colors of cyan, magenta, and yellow with labels for printer manufacturer and model."----

# Specify the figure font size
f9_fs = 46

# Make list of printer manufacturers and models
inkjet_printer_types <-
  data.frame(
    "printer" = 1:11,
    "manufacturer_model" = c(
      "Brother \nMFC- \n665CW",
      "Canon \nPixma \nMX340",
      "Canon PG \n210XL \n",
      "Epson \nUnknown \n",
      "HP \nOfficejet \n5740",
      "HP \nDeskjet \nf5180",
      "HP \nOfficejet \n6500",
      "HP \nOfficejet \n6500",
      "Lexmark \n228 2010 \nCE 81",
      "Sensient \nUnknown \n",
      "Sensient \nUnknown \n"
    )
  )

# Join printer info
inkjet_printer_plot_data <-
  inkjet |>
    mutate(
      color = fct_recode(color, "Cyan" = "c", "Magenta" = "m", "Yellow" = "y"),
      spectra = -spectra
    ) |>
    left_join(inkjet_printer_types) |>
    mutate(printer = paste0("Printer ", printer, " \n", manufacturer_model))

# Re-level factor
inkjet_printer_plot_data$printer <- 
  factor(
    x = inkjet_printer_plot_data$printer, 
    levels = unique(inkjet_printer_plot_data$printer)
  )

# Create figure
inkjet_printer_plot_data |>
  ggplot(aes(x = spectra, y = intensity, group = sample, color = color)) +
  geom_line(alpha = 0.9, linewidth = 0.7) +
  facet_grid(color ~ printer) +
  labs(x = TeX("Wavenumber (cm$^{-1})$"), y = "Intensity") +
  scale_color_manual(values = col_inkjet) +
  scale_x_reverse() +
  theme_bw(base_size = 26) +
  theme(
    legend.position = "none",
    strip.background = element_blank(),
    panel.spacing = unit(0.25, "lines"),
    axis.title = element_text(size = f9_fs),
    axis.text = element_text(size = f9_fs*0.7),
    title = element_text(size = f9_fs),
    strip.text = element_text(size = f9_fs*0.75),
    legend.text = element_text(size = f9_fs),
    legend.title = element_text(size = f9_fs),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)
  )


## ----tab2----------------------------------------------------------------------------------------------------------------
inkjet_res_best |>
  select(color, s, pcs, ntrees, acc_ave) |>
  rename(veesa_acc = acc_ave) |>
  mutate(scenario = "Best") |>
  bind_rows(
    inkjet_res_worst |>
    select(color, s, pcs, ntrees, acc_ave) |>
    rename(veesa_acc = acc_ave) |>
    mutate(scenario = "Worst")
  ) |>
  left_join(inkjet_paper_res_best |> select(color, acc_ave), by = "color") |>
  select(scenario, everything()) |>
  mutate(acc_ave = ifelse(scenario == "Worst", "-", acc_ave)) |>
  rename(
    Scenario = scenario,
    Color = color,
    "Box Filter" = s,
    "PCs" = pcs,
    "Trees" = ntrees,
    "VEESA" = veesa_acc,
    "Buzzini" = acc_ave
  ) |>
  knitr::kable(
    caption = "Inkjet Data Cross Validation Average Accuracies.",
    booktabs = T
  )


## ----fig9, fig.height = 8.5, fig.width = 20, out.width = "6in", fig.cap = "\\emph{Inkjet Data Feature Importances.} Boxplots of PFI values across 10 replications from the best and worst performing models for cyan, magenta, and yellow inkjet signatures."----

# Specify the figure font size
f11_fs = 28

bind_rows(
  map_df(inkjet_pfi_best, "pfi_df") |> mutate(scenario = "Best Model"),
  map_df(inkjet_pfi_worst, "pfi_df") |> mutate(scenario = "Worst Model")
) |>
  pivot_longer(cols = pfi_reps.1:pfi_reps.10, names_to = "rep", values_to = "fi") |>
  group_by(color, pc, scenario) |>
  mutate(
    fi_min = min(fi),
    fi_max = max(fi)
  ) |>
  ungroup() |>
  ggplot(aes(x = pc, y = fi, group = pc)) +
  geom_boxplot(aes(fill = color, color = color), alpha = 0.5) +
  facet_grid(color ~ scenario, scales = "free_x", space = "free_x") +
  scale_color_manual(values = col_inkjet) +
  scale_fill_manual(values = col_inkjet) +
  theme_bw(base_size = f11_fs, base_family = ff) +
  theme(
    legend.position = "none",
    strip.background = element_rect(color = "white", fill = "white"),
    axis.title = element_text(size = f11_fs),
    axis.text = element_text(size = f11_fs*0.7),
    title = element_text(size = f11_fs),
    strip.text = element_text(size = f11_fs),
    legend.text = element_text(size = f11_fs),
    legend.title = element_text(size = f11_fs)
  )  +
  labs(
    x = "Principal component",
    y = "Feature importance"
  )



## ----fig10, fig.height = 13.5, fig.width = 25, out.width = "5.75in", fig.cap = "\\emph{Inkjet Data Principal Direction Plots.} jfPCs with the largest PFI values from the best (top) and worst (bottom) models for predicting cyan inkjet signatures. jfPCs ordered from left to right based on highest to lowest feature importance."----

# Specify the figure font size
f12_fs = 30

plot_inkjet_pcs <- function(pfi_res) {
  pc_order <-
    pfi_res$pfi_df |>
    arrange(desc(pfi)) |>
    pull(pc)
  veesa::plot_pc_directions(
    fpc = pc_order[1:5],
    fdasrvf = pfi_res$jfpca,
    fpca_method = "jfpca",
    times = -unique(inkjet$spectra),
    alpha = 0.9,
    nrow = 1,
    linesizes = rep(1.5, 7),
    linetype = "solid"
  ) +
    scale_x_reverse() +
    theme_bw(base_size = f12_fs, base_family = ff) +
    theme(
      legend.position = "bottom",
      strip.background = element_rect(color = "white", fill = "white"),
      axis.title = element_text(size = f12_fs),
      axis.text = element_text(size = f12_fs),
      title = element_text(size = f12_fs),
      strip.text = element_text(size = f12_fs),
      legend.text = element_text(size = f12_fs),
      legend.title = element_text(size = f12_fs)
    )  +
    scale_color_manual(values = col_pcdir_3sd) +
    guides(
      color = guide_legend(nrow = 1),
      size = guide_legend(nrow = 1),
      override.aes = list(alpha = 1, size = 2)
    ) +
    labs(
      x = TeX("Wavenumber (cm$^{-1})$"),
      y = "Intensity",
      title = paste(pfi_res$pfi_df$color[1])
    )
}

inkjet_cyan_best_pc_dirs <-
  plot_inkjet_pcs(inkjet_pfi_best[[1]]) + 
  labs(title = "Cyan (best)") + 
  guides(color = guide_legend(override.aes = list(linewidth = 2)))

inkjet_cyan_worst_pc_dirs <-
  plot_inkjet_pcs(inkjet_pfi_worst[[1]]) +
  labs(title = "Cyan (worst)") + 
  guides(color = guide_legend(override.aes = list(linewidth = 2)))

plot_grid(
  inkjet_cyan_best_pc_dirs + theme(legend.position = "none"),
  inkjet_cyan_worst_pc_dirs + theme(legend.position = "none"),
  get_plot_component(
    inkjet_cyan_best_pc_dirs + theme(legend.key.width = unit(2, 'cm')),
    'guide-box-bottom',
    return_all = TRUE
  ),
  ncol = 1,
  rel_heights = c(0.3, 0.3, 0.05)
)

