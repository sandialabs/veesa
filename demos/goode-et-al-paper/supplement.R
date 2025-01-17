## ----setup, include = FALSE------------------------

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
library(forcats)
library(ggplot2)
library(latex2exp)
library(purrr)
library(randomForest)
library(reticulate)
library(stringr)
library(tidyr)
library(veesa)
library(wesanderson)

# Specify the conda environment to use
use_condaenv(condaenv = "veesa", required = TRUE)

# Specify colors for groups
col_2groups = wes_palettes$Royal1[2:1]

# Specify colors for PC direction plots
col_plus1 = "#784D8C"
col_plus2 = "#A289AE"
col_plus3 = "#CAC0D2"
col_minus1 = "#EA9B44"
col_minus2 = "#EBBC88"
col_minus3 = "#F3DABC"
col_pcdir_2sd = c(col_plus2, col_plus1, "black", col_minus1, col_minus2)
col_pcdir_3sd = c(col_plus3, col_plus2, col_plus1, "black", col_minus1, col_minus2, col_minus3)
 
# Specify inkjet colors
col_inkjet = c("#0EB4AD", "#8f1883", "#d9c004")

# Specify some high/low colors for contrast throughout
col_low = wes_palette(name = "Royal1", n = 30, type = "continuous")[4]
col_high = wes_palette(name = "Royal1", n = 30, type = "continuous")[29]

# Specify the font (ff) for all plots
ff = "Helvetica"

# Specify a file path
fp = "~/OneDrive - Sandia National Laboratories/Documents/projects/veesa/"  # kjgoode
# fp = "~/OneDrive - Sandia National Laboratories/veesa/"  # jdtuck
# fp = "~/veesa/"


## ----shifted-peaks-data----------------------------
# Load the sim data
shifted_peaks_data = shifted_peaks$data |> mutate(group = factor(group))

# Create long versions of training/testing data sets
shifted_peaks_train_long = shifted_peaks_data %>% filter(data == "Training")
shifted_peaks_test_long = shifted_peaks_data %>% filter(data == "Testing")

# Create wide versions of training and testing data
shifted_peaks_train_wide <- 
  shifted_peaks_train_long %>%
  select(-t) %>%
  mutate(index = paste0("t", index)) %>%
  pivot_wider(names_from = index, values_from = y)
shifted_peaks_test_wide <- 
  shifted_peaks_test_long %>%
  select(-t) %>%
  mutate(index = paste0("t", index)) %>%
  pivot_wider(names_from = index, values_from = y)

# Determine number of observations in training/testing data
shifted_peaks_train_nobs = dim(shifted_peaks_train_wide)[1]
shifted_peaks_test_nobs = dim(shifted_peaks_test_wide)[1]

# Determine the time points where values are observed
shifted_peaks_times = sort(unique(shifted_peaks_train_long$t))

# Load ESA version of the train/test data
shifted_peaks_train_esa_fp = paste0(fp, "data/shifted-peaks/shifted-peaks-train-esa.rds")
shifted_peaks_train_esa = readRDS(shifted_peaks_train_esa_fp)
shifted_peaks_test_esa_fp = paste0(fp, "data/shifted-peaks/shifted-peaks-test-esa.rds")
shifted_peaks_test_esa = readRDS(shifted_peaks_test_esa_fp)
  
# Prepare data for model
shifted_peaks_train_model_data <- 
  shifted_peaks_train_esa$fpca_res$coef %>% 
  data.frame() %>% 
  mutate(group = factor(shifted_peaks_train_wide$group))
shifted_peaks_test_model_data <- 
  t(shifted_peaks_test_esa$coef) %>% 
  data.frame() %>% 
  mutate(group = factor(shifted_peaks_test_wide$group))

# Specify the number of replications used to compute PFI with the sim data
shifted_peaks_K = 10


## ----shifted-peaks-true-means----------------------
# # Specify parameters
# z1 = 1; z1_sd = 0.05; a1 = -3; a1_sd = 1
# z2 = 1.25; z2_sd = 0.05; a2 = 3; a2_sd = 1
# 
# # Function for generating true means
# true_mean = function(group, t) {
#   if (group == 1) {
#    z = z1
#    a = a1 
#   }
#   if (group == 2) {
#     z = z2
#     a = a2
#   }
#   y = z * exp((-((t - a)^2)) / 2)
#   data.frame(group = as.character(group), index = 1:length(t), t, y)
# }

# True functional means
shifted_peaks_true_means <- shifted_peaks$true_means
  # bind_rows(true_mean(1, shifted_peaks_times), true_mean(2, shifted_peaks_times)) %>% 
  # select(-index) %>%
  # mutate(group = factor(group, levels = c("1", "2")), t = round(t,2)) %>%
  # rename(mean_true = y)


## ----shifted-peaks-cross-sectional-----------------
# Prepare cross-sectional data
shifted_peaks_data_cs_train = shifted_peaks_train_wide %>% select(-id, -data) %>% mutate(group = factor(group))
shifted_peaks_data_cs_test = shifted_peaks_test_wide %>% select(-id, -data) %>% mutate(group = factor(group))

# cross-sectional approach
shifted_peaks_rf_cs_fp = paste0(fp, "results/shifted-peaks/shifted-peaks-rf-cross-sectional.rds")
if (file.exists(shifted_peaks_rf_cs_fp)) {
  shifted_peaks_rf_cs = readRDS(shifted_peaks_rf_cs_fp)
} else {
  set.seed(20210301)
  shifted_peaks_rf_cs <-
    randomForest::randomForest(
      group ~ .,
      shifted_peaks_data_cs_train,
      importance = TRUE
    )
  saveRDS(shifted_peaks_rf_cs, shifted_peaks_rf_cs_fp)
}

# Compute the model predictions on train/test data
shifted_peaks_pred_cs_train = predict(shifted_peaks_rf_cs, shifted_peaks_data_cs_train)
shifted_peaks_pred_cs_test = predict(shifted_peaks_rf_cs, shifted_peaks_data_cs_test)

# Compute the accuracy on train/test data
shifted_peaks_acc_cs_train = sum(shifted_peaks_pred_cs_train == shifted_peaks_train_wide$group) / shifted_peaks_train_nobs
shifted_peaks_acc_cs_test = sum(shifted_peaks_pred_cs_test == shifted_peaks_test_wide$group) / shifted_peaks_test_nobs

# Compute PFI using accuracy for the cross-sectional approach
shifted_peaks_pfi_acc_cs_fp = paste0(fp, "results/shifted-peaks/shifted-peaks-pfi-acc-cross-sectional.rds")
if (file.exists(shifted_peaks_pfi_acc_cs_fp)) {
  shifted_peaks_pfi_acc_cs = readRDS(shifted_peaks_pfi_acc_cs_fp)
} else {
  set.seed(20210921)
  shifted_peaks_pfi_acc_cs <- 
    compute_pfi(
      x = shifted_peaks_data_cs_test[, -1],
      y = shifted_peaks_data_cs_test$group,
      f = shifted_peaks_rf_cs,
      K = shifted_peaks_K, 
      metric = "accuracy"
    )
  saveRDS(shifted_peaks_pfi_acc_cs, shifted_peaks_pfi_acc_cs_fp)
}

# Compute PFI using log-loss for the cross-sectional approach
shifted_peaks_pfi_ll_cs_fp = paste0(fp, "results/shifted-peaks/shifted-peaks-pfi-ll-cross-sectional.rds")
if (file.exists(shifted_peaks_pfi_ll_cs_fp)) {
  shifted_peaks_pfi_ll_cs = readRDS(shifted_peaks_pfi_ll_cs_fp)
} else {
  set.seed(20210921)
  shifted_peaks_pfi_ll_cs <- 
    compute_pfi(
      x = shifted_peaks_data_cs_test[, -1],
      y = shifted_peaks_data_cs_test$group,
      f = shifted_peaks_rf_cs,
      K = shifted_peaks_K, 
      metric = "logloss"
    )
  saveRDS(shifted_peaks_pfi_ll_cs, shifted_peaks_pfi_ll_cs_fp)
}


## ----shifted-peaks-jfpca---------------------------
# Random forest on jfPCA with aligned data
shifted_peaks_rf_jfpca_fp = paste0(fp, "results/shifted-peaks/shifted-peaks-rf-jfpca.rds")
shifted_peaks_rf_jfpca = readRDS(shifted_peaks_rf_jfpca_fp)

# Compute the model predictions on the test data
shifted_peaks_train_pred_jfpca = predict(shifted_peaks_rf_jfpca, shifted_peaks_train_model_data %>% select(-group))
shifted_peaks_test_pred_jfpca = predict(shifted_peaks_rf_jfpca, shifted_peaks_test_model_data %>% select(-group))

# Compute the accuracy on the training data
shifted_peaks_train_acc_jfpca = sum(shifted_peaks_train_pred_jfpca == shifted_peaks_train_model_data$group) / shifted_peaks_train_nobs
shifted_peaks_test_acc_jfpca = sum(shifted_peaks_test_pred_jfpca == shifted_peaks_test_model_data$group) / shifted_peaks_test_nobs

# Compute PFI using log-loss with the VEESA pipeline
shifted_peaks_pfi_ll_jfpca_fp = paste0(fp, "results/shifted-peaks/shifted-peaks-pfi-ll-jfpca.rds")
if (file.exists(shifted_peaks_pfi_ll_jfpca_fp)) {
  shifted_peaks_pfi_ll_jfpca = readRDS(shifted_peaks_pfi_ll_jfpca_fp)
} else {
  set.seed(20210921)
  shifted_peaks_pfi_ll_jfpca <-
    compute_pfi(
      x = shifted_peaks_test_model_data %>% select(-group),
      y = shifted_peaks_test_model_data$group,
      f = shifted_peaks_rf_jfpca,
      K = shifted_peaks_K,
      metric = "logloss"
    )
  saveRDS(shifted_peaks_pfi_ll_jfpca, shifted_peaks_pfi_ll_jfpca_fp)
}

# Load aligned PC directions
shifted_peaks_jfpca_aligned_fp = paste0(fp, "data/shifted-peaks/shifted-peaks-jfpca-aligned.rds")
shifted_peaks_jfpca_aligned = readRDS(shifted_peaks_jfpca_aligned_fp)


## ----shifted-peaks-check---------------------------
if (data.frame(shifted_peaks_pfi_acc_cs$pfi_single_reps) %>% summarise_all(.funs = sd) %>% as.numeric() %>% unique() > 0)
  stop("Some times have PFI standard deviation > 0.")


## ----shifted-peaks-pfi-cutoff----------------------
shifted_peaks_veesa_pfi_cutoff = 0.02


## ----figS1, fig.width = 6, out.width = '4in', fig.cap = "Spearman correlations between all pairs of cross-sectional variables from the simulated data."----

# Specify the figure size (for determining font size)
fS1_fs = 14

# Prepare the correlations for the plot
corr_data <-
  shifted_peaks_train_wide %>%
  select(-id, -group, -data) %>%
  cor(method = "spearman") %>%
  as.data.frame() %>%
  mutate(index_row = 1:n()) %>%
  reshape2::melt(id = "index_row") %>%
  rename(index_col = variable) %>%
  mutate(index_row = as.numeric(str_remove(index_row, "t")),
         index_col = as.numeric(str_remove(index_col, "t"))) %>%
  left_join(shifted_peaks_data %>%
              select(index, t) %>%
              distinct(),
            by = c("index_row" = "index")) %>%
  rename(t_row = t) %>%
  left_join(shifted_peaks_data %>%
              select(index, t) %>%
              distinct(),
            by = c("index_col" = "index")) %>%
  rename(t_col = t) %>%
  select(index_col, t_col, index_row, t_row, value)

# Plot the correlation matrix
corr_data %>%
  ggplot(aes(x = t_col, y = t_row)) +
  geom_tile(aes(fill = value, color = value)) +
  coord_equal() +
  guides(color = guide_colourbar(barheight = 4, barwidth = 0.5)) +
  scale_color_gradient2(
    low = col_low,
    high = col_high,
    mid = "white",
    limits = c(-1, 1)
  ) +
  scale_fill_gradient2(
    low = col_low,
    high = col_high,
    mid = "white",
    limits = c(-1, 1)
  ) +
  guides(color = guide_colourbar(barheight = 6, barwidth = 1)) +
  labs(
    x = TeX('$t$'),
    y = TeX('$t$'),
    fill = "Correlation",
    color = "Correlation"
  ) +
  theme_bw(base_family = ff, base_size = fS1_fs) +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.ticks = element_blank(),
    aspect.ratio = 1
  )


## ----figS2, fig.width = 15, fig.height = 10, out.width = '6.5in', fig.cap = "(A) The solid lines represent the true group means. The dots and error bars represent the cross-sectional group means plus and minus one cross-sectional standard deviation. (B-D) PFI values computed on the test data using the metric of accuracy for a random forest trained using the cross-sectional approach, using the metric of log-loss for a random forest trained using the cross-sectional approach, and the metric of log-loss for a random forest trained using the VEESA pipeline, respectively."----

# Specify the figure size (for determining font size)
fS2_fs = 16

# Specify limits associated with the region with high 
# importance from the cross-sectional approach
f1_ll = -0.7
f1_ul = 3.5

# Specify an aspect ratio for all plots relating to the sim data
shifted_peaks_ar = 0.5

# Plot of cross-sectional means and standard deviations by group
plot_shifted_peaks_cs_means <-
  shifted_peaks_data %>%
  group_by(t, group) %>%
  summarise(mean_y = mean(y),
            sd_y = sd(y),
            .groups = "drop") %>%
  ggplot(aes(x = t, y = mean_y)) +
  geom_errorbar(aes(
    ymin = mean_y - sd_y,
    ymax = mean_y + sd_y,
    color = group
  ), alpha = 0.6) +
  geom_point(aes(color = group)) +
  geom_line(
    data = shifted_peaks_true_means,
    mapping = aes(
      x = t,
      y = mean_true,
      group = group,
      color = group
    ),
    size = 1.5,
    alpha = 0.5
  ) +
  labs(
    title = "Cross-sectional means plus/minus one cross-sectional \nstandard deviation",
    y = "y",
    x = TeX('$t$'),
    y = TeX('$y_{g,i}(t)$'),
    color = "Group",
    fill = "Group"
  ) +
  scale_color_manual(values = col_2groups) +
  scale_fill_manual(values = col_2groups) +
  theme_bw(base_family = ff, base_size = fS2_fs) +
  theme(legend.position = "bottom", aspect.ratio = shifted_peaks_ar) +
  ylim(-0.15, 1.4)

# Plot the PFI from the cross-sectional approach (using accuracy as the metric)
plot_cs_pfi_acc <-
  data.frame(
    time = sort(unique(shifted_peaks_data$t)),
    imp = shifted_peaks_pfi_acc_cs$pfi
  ) %>%
  ggplot(aes(x = time, y = imp)) +
  geom_point() +
  geom_segment(aes(xend = time, yend = 0)) +
  labs(
    title = "Feature importance for cross-sectional model \n(computed using log-loss)",
    x = TeX('$t$'),
    y = "PFI"
  ) +
  theme_bw(base_family = ff, base_size = fS2_fs) + 
  theme(aspect.ratio = shifted_peaks_ar)

# Plot the PFI from the cross-sectional approach (using log-loss as the metric)
plot_cs_pfi_ll <-
  data.frame(
    time = sort(unique(shifted_peaks_data$t)),
    imp = shifted_peaks_pfi_ll_cs$pfi
  ) %>%
  ggplot(aes(x = time, y = imp)) +
  geom_point() +
  geom_segment(aes(xend = time, yend = 0)) +
  labs(
    title = "Feature importance for cross-sectional model \n(computed using log-loss)",
    x = TeX('$t$'),
    y = "PFI"
  ) +
  theme_bw(base_family = ff, base_size = fS2_fs) + 
  theme(aspect.ratio = shifted_peaks_ar)

# Plot the PFI from the VEESA pipeline (using log-loss as the metric)
plot_veesa_pfi_ll <-
  data.frame(fpc = 1:length(shifted_peaks_pfi_ll_jfpca$pfi),
             imp = shifted_peaks_pfi_ll_jfpca$pfi) %>%
  ggplot(aes(x = fpc, y = imp)) +
  geom_point() +
  geom_segment(aes(xend = fpc, yend = 0)) +
  labs(title = "Feature importance for VEESA pipeline model \n(computed using log-loss)",
       x = TeX('jfPC'),
       y = "PFI") +
  theme_bw(base_family = ff, base_size = fS2_fs) + 
  theme(aspect.ratio = shifted_peaks_ar) + 
  ylim(0, 0.15)

# Extract the legend
legend <- 
  get_plot_component(
    plot_shifted_peaks_cs_means, 
    'guide-box-bottom', 
    return_all = TRUE
  )

# Join the plots
plot_grid(
  plot_grid(
    plot_shifted_peaks_cs_means + 
      theme(legend.position = "none"),
    plot_cs_pfi_acc,
    plot_cs_pfi_ll,
    plot_veesa_pfi_ll,
    ncol = 2,
    labels = c("A", "B", "C", "D")
  ),
  legend,
  ncol = 1,
  rel_heights = c(0.95, 0.05)
)


## ----figS3, fig.width = 16, fig.height = 12, out.width = "6.5in", fig.cap = paste0("Principal direction plots of the VEESA pipeline jfPCs with log-loss based PFI values greater than ", shifted_peaks_veesa_pfi_cutoff, " (excluding jfPCs 1 and 2 shown in Figures 2 and 4).")----

# Specify the figure size (for determining font size)
fS3_fs = 18.5

shifted_peaks_veesa_pc_top <- 
  data.frame(pfi = shifted_peaks_pfi_ll_jfpca$pfi) %>%
  mutate(pc = 1:n()) %>%
  filter(pfi > shifted_peaks_veesa_pfi_cutoff, pc != 1, pc !=2) %>%
  arrange(desc(pfi)) %>%
  pull(pc)

plot_pc_directions(
  fpc = shifted_peaks_veesa_pc_top,
  fdasrvf = shifted_peaks_jfpca_aligned$fpca_res,
  fpca_method = "jfpca",
  times = -shifted_peaks_times,
  alpha = 0.9, 
  nrow = 3, 
  linesizes = c(0.75, 1, 1.25, 1, 0.75), 
  linetype = FALSE
) +
  theme_bw(base_family = ff, base_size = fS3_fs) +
  theme(
    strip.background = element_rect(color = "white", fill = "white"),
    legend.key.width = unit(1,"cm"),
    aspect.ratio = 0.4,
    legend.position = "bottom",
    axis.title = element_text(size = fS3_fs), 
    axis.text = element_text(size = fS3_fs), 
    title = element_text(size = fS3_fs),
    strip.text = element_text(size = fS3_fs),
    legend.text = element_text(size = fS3_fs),
    legend.title = element_text(size = fS3_fs)
  ) +
  scale_color_manual(values = col_pcdir_2sd) +
  guides(color = guide_legend(override.aes = list(linewidth = 1.25))) +
  labs(x = TeX('$t$'), y = TeX('$y_g(t)$'))


## ----hct-load-res----------------------------------

# Load subset of training data
hct_train_sub = py_load_object(paste0(fp, "data/hct/hct-sub-train.pkl"))

# Load principal components from training data
hct_train_jfpca_pcs = py_load_object(paste0(fp, "data/hct/hct-train-jfpca-centered-pc-dirs-sparam15.pkl"))

# Load latent values from training data
hct_train_jfpca_latent = py_load_object(paste0(fp, "data/hct/hct-train-jfpca-latent-sparam15.pkl"))
hct_train_vfpca_latent = py_load_object(paste0(fp, "data/hct/hct-train-vfpca-latent-sparam15.pkl"))

# Load PFI results
hct_jfpca_nn_pfi = py_load_object(paste0(fp, "results/hct/hct-test-pfi-jfpca-sparam15.pkl"))
hct_vfpca_nn_pfi = py_load_object(paste0(fp, "results/hct/hct-test-pfi-vfpca-sparam15.pkl"))

# Vector of smoothing parameter values
sparams = c("01", "05", "10", "15", "20", "25")

# Function for loading predictions and accuracies
load_results <- function(train_test, result, stage, sparam) {
  fp = paste0(
    fp, "results/hct/hct-", train_test, "-", 
    result, "-", stage, "-sparam", sparam, ".pkl"
  )
  py_load_object(fp)
}

# Load predictions and accuracies
hct_pred_and_metrics_train <- 
  set_names(c("jfpca", "vfpca", "hfpca")) %>%
  map(.f = function(x) set_names(sparams) %>%
  map(
    .f = load_results, 
    stage = x, 
    train_test = "train", 
    result = "pred-and-metrics")
  )

hct_pred_and_metrics_test <- 
  set_names(c("jfpca", "vfpca", "hfpca")) %>%
  map(.f = function(x) set_names(sparams) %>%
  map(
    .f = load_results, 
    stage = x, 
    train_test = "test", 
    result = "pred-and-metrics")
)

hct_pred_and_metrics_cs_pre_smooth <- 
  set_names(c("train", "test")) %>%
  map(
    .f = load_results, 
    stage = "cs-pre-smooth",
    result = "pred-and-metrics", 
    sparam = "00"
  )

hct_pred_and_metrics_cs_pre_align <- 
  set_names(c("train", "test")) %>%
  map(.f = function(x) set_names(sparams) %>%
  map(
    .f = load_results, 
    stage = "cs-pre-align", 
    train_test = x, 
    result = "pred-and-metrics")
)

hct_pred_and_metrics_cs_post_align <- 
  set_names(c("train", "test")) %>%
  map(.f = function(x) set_names(sparams) %>%
  map(
    .f = load_results, 
    stage = "cs-post-align", 
    train_test = x, 
    result = "pred-and-metrics")
  )



## ----hct-cv-metrics--------------------------------

# Prepare metrics for plotting 
hct_pred_and_metrics_train_df <-
  map_df(
    .x = hct_pred_and_metrics_train,
    .f = function(x)
      map_df(x, "acc"),
    .id = "pca"
  ) %>%
  mutate(data = "training")

hct_pred_and_metrics_test_df <-
  map_df(
    .x = hct_pred_and_metrics_test,
    .f = function(x)
      map_df(x, "acc"),
    .id = "pca"
  ) %>% 
  mutate(data = "testing")

hct_pred_and_metrics_cs_pre_smooth_df <-
  map_df(
    .x = hct_pred_and_metrics_cs_pre_smooth,
    .f = "acc"
  ) %>% 
  mutate(pca = "CS pre-smooth") %>%
  pivot_longer(cols = -pca, names_to = "data", values_to = "00") %>%
  mutate(data = ifelse(data == "train", "training", "testing"))

hct_pred_and_metrics_cs_pre_align_df <-
  map_df(
    .x = hct_pred_and_metrics_cs_pre_align,
    .f = function(x)
      map_df(x, "acc"),
    .id = "data"
  ) %>% 
  mutate(data = ifelse(data == "train", "training", "testing")) %>%
  mutate(pca = "CS pre-align")

hct_pred_and_metrics_cs_post_align_df <-
  map_df(
    .x = hct_pred_and_metrics_cs_post_align,
    .f = function(x)
      map_df(x, "acc"),
    .id = "data"
  ) %>% 
  mutate(data = ifelse(data == "train", "training", "testing")) %>%
  mutate(pca = "CS post-align")



## ----hct-pfi-pcs-----------------------------------
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

# Join PFI for jfPCA and vfPCA
hct_pfi <-
  bind_rows(
    data.frame(hct_jfpca_nn_pfi$importances) |>
      mutate(fpca = "Joint fPCA", fpc = 1:n()),
    data.frame(hct_vfpca_nn_pfi$importances) |>
      mutate(fpca = "Vertical fPCA", fpc = 1:n())
  ) |>
  pivot_longer(
    cols = X1:X5,
    values_to = "fi",
    names_to = "rep"
  ) |>
  mutate(rep = str_remove(rep, "X"))


## ----figS4, fig.width = 10, fig.height = 5, out.width = '6.5in', fig.cap = "Model accuracies from neural networks applied using the VEESA pipeline and the cross-sectional approach. "----

# Specify the figure size (for determining font size)
fS4_fs = 15

# Create plot
bind_rows(
  hct_pred_and_metrics_train_df,
  hct_pred_and_metrics_test_df,
  hct_pred_and_metrics_cs_pre_smooth_df,
  hct_pred_and_metrics_cs_pre_align_df,
  hct_pred_and_metrics_cs_post_align_df
) %>%
  pivot_longer(cols = -c(data, pca),
               names_to = "sparam",
               values_to = "acc") %>%
  mutate(sparam = as.numeric(sparam)) %>%
  mutate(pca = forcats::fct_relevel(
    pca,
    c(
      "CS pre-smooth",
      "CS pre-align",
      "CS post-align",
      "jfpca",
      "vfpca",
      "hfpca"
    )
  )) %>%
  ggplot(aes(
    x = sparam,
    y = acc,
    color = pca,
    linetype = data,
    shape = data,
    group = factor(data):factor(pca)
  )) +
  geom_point(size = 2) +
  geom_line(size = 0.75) +
  theme_bw(base_family = ff, base_size = fS4_fs) +
  gretchenalbrecht::scale_color_gretchenalbrecht(palette = "last_rays") +
  labs(
    x = "Number of Box-Filter Runs",
    y = "Accuracy",
    color = "Method",
    linetype = "Dataset",
    shape = "Dataset"
  ) + 
  ylim(0.5,1)


## ----figS5, fig.width = 12, fig.height = 8, out.width = '5.5in', fig.cap = "PFI replicate values for jfPCA (top) and vfPCA (bottom) associated with the H-CT material classification example.", warning = FALSE----

# Specify the figure font size
fS5_fs = 20

# Create plots of proportion of variation explained and PFI values
hct_pfi |>
  ggplot(aes(x = fpc, y = fi, color = rep)) +
  geom_point(size = 2, alpha = 0.5) +
  facet_grid(fpca ~ ., switch = "y") +
  labs(x = "Principal Component", color = "PFI replicate") +
  theme_bw(base_family = ff, base_size = fS5_fs) +
  theme(
    strip.placement = "outside",
    strip.background = 
      element_rect(color = "white", fill = "white"),
    axis.title.y = element_blank(),
    axis.title.x = element_text(size = fS5_fs),
    axis.text = element_text(size = fS5_fs),
    title = element_text(size = fS5_fs),
    strip.text = element_text(size = fS5_fs),
    legend.position = "bottom"
  ) + 
  scale_color_manual(values = wes_palettes$Zissou1[5:1])


## ----figS6, fig.width = 16, fig.height = 12, out.width = "6.5in", fig.cap = "Principal directions from the six jfPCs with the highest PFI from the H-CT material classification example."----

# Specify the figure font size
fS6_fs = 18.5

plot_pc_directions(
  fpcs = hct_top_jfpcs,
  fdasrvf = list(
    "f_pca" = hct_train_jfpca_pcs[, 2:6, ],
    "latent" = hct_train_jfpca_latent
  ),
  times = unique(hct_train_sub$frequency_norm),
  fpca_method = "jfpca",
  alpha = 0.9,
  nrow = 3,
  linesizes = rep(0.75, 5),
  linetype = FALSE
) +
  theme_bw(base_size = fS6_fs, base_family = ff) +
  theme(
    legend.key.width = unit(1, "cm"),
    strip.background = element_rect(color = "white", fill = "white"),
    aspect.ratio = 0.4,
    legend.position = "bottom",
    axis.title = element_text(size = fS6_fs), 
    axis.text = element_text(size = fS6_fs), 
    title = element_text(size = fS6_fs),
    strip.text = element_text(size = fS6_fs),
    legend.text = element_text(size = fS6_fs),
    legend.title = element_text(size = fS6_fs)
  )  +
  scale_color_manual(values = col_pcdir_2sd) +
  guides(color = guide_legend(override.aes = list(linewidth = 1.25))) +
  labs(
    x = "Normalized Frequency",
    y = "Intensity",
    color = "Material",
    size = "Material",
    linetype = "Material"
  )


## ----inkjet-load-data-and-res----------------------

# Load inkjet data
inkjet = read.csv(paste0(fp, "data/inkjet/inkjet-cleaned.csv"))

# Load cross validation folds 
inkjet_folds = read.csv(paste0(fp, "data/inkjet/inkjet-cv-folds.csv"))

# Load CV results
inkjet_cv_preds <-
  readRDS(paste0(fp, "results/inkjet/inkjet-cv-preds.rds")) |>
  mutate(
    color = fct_recode(
      color, "Cyan" = "c", "Magenta" = "m", "Yellow" = "y"
    )
  )

# Load CV results from Buzzini paper
inkjet_paper_res_best = read.csv(paste0(fp, "results/inkjet/inkjet-paper-res-best.csv"))

# Load cross validation results
inkjet_cv_acc_summary = read.csv(paste0(fp, "results/inkjet/inkjet-cv-accuracy-summary.csv"))

# Load PFI results
inkjet_pfi_best = py_load_object(paste0(fp, "results/inkjet/inkjet-pfi-best.csv"), convert = TRUE)
inkjet_pfi_worst = py_load_object(paste0(fp, "results/inkjet/inkjet-pfi-worst.pkl"), convert = TRUE)

# Load CV scenarios
inkjet_smooth_rep_fold_color_list = readRDS(paste0(fp, "results/inkjet/inkjet-cv-srfc.rds"))



## ----inkjet-improve-worst--------------------------

# Specify a file path
inkjet_cv_preds_pfi_informed_fp = paste0(fp, "results/inkjet/inkjet-cv-preds-pfi-informed.rds")

# Train models with best performing PCs
if (!file.exists(inkjet_cv_preds_pfi_informed_fp)) {

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
  
  # Function for implementing cross-validation for inkjet data
  get_inkjet_rf_cv_preds_pfi_informed <- function(s, rep_curr, test_fold, color) {
  
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

    # Create a dataframe with training data
    rf_data <- data.frame(
      printer = factor(train_printer_sample$printer), 
      train$coef[,c(1:25, 75:100)]
    )
    
    # Train random forest
    set.seed(20230630)
    rf = randomForest(printer ~ ., data = rf_data, ntree = 500)
  
    # Get test fold predictions
    preds = predict(rf, data.frame(test$coef[,c(1:25, 75:100)]))
    
    # Put predictions in a dataframe
    preds_df <-
      data.frame(
        color = color,
        s = s,
        rep = rep_curr,
        test_fold = test_fold,
        printer = test_printer_sample$printer,
        sample = test_printer_sample$sample, 
        pred = as.numeric(as.vector(preds))
      )
    
    # Return the results
    return(preds_df)
      
  }
  
  # Implement CV for all specified scenarios
  inkjet_cv_preds_pfi_informed <- 
    pmap_df(
      .l = inkjet_smooth_rep_fold_color_list,
      .f = get_inkjet_rf_cv_preds_pfi_informed
    )
  
  # Save PFI informed model predictions
  saveRDS(inkjet_cv_preds_pfi_informed, inkjet_cv_preds_pfi_informed_fp)
  
}

# Load results
inkjet_cv_preds_pfi_informed = readRDS(inkjet_cv_preds_pfi_informed_fp)

# Compute test fold accuracy
inkjet_cv_acc_pfi_informed <-
  inkjet_cv_preds_pfi_informed |>
  summarise(
    acc = sum(pred == printer) / length(printer),
    .by = c(s, rep, test_fold, color)
  )

# Compute summary values of CV accuracies
inkjet_cv_acc_pfi_informed_summary <-
  inkjet_cv_acc_pfi_informed |>
  summarise(
    acc_ave = mean(acc),
    acc_sd = sd(acc),
    acc_min = min(acc),
    acc_max = max(acc), 
    .by = c(s, color)
  ) |>
  mutate(color = factor(color)) |>
  mutate(color = fct_recode(color, "Cyan" = "c", "Magenta" = "m", "Yellow" = "y"))



## ----figS7, fig.height = 18, fig.width = 18, out.width = "5.25in", fig.cap = "Confusion matrices for the inkjet printer random forests."----

# Specify the figure font size
fS7_fs = 24

inkjet_plot_conf_cyan <-
  inkjet_cv_preds |>
  filter(
    color == "Cyan",
    s == 20, 
    pcs == 40,
    ntrees == 1000
  ) |>
  count(printer, pred, name = "n_preds") |>
  ggplot(aes(x = printer, y = pred, fill = n_preds)) +
  geom_tile() +
  geom_label(aes(label = n_preds), size = 6) +
  scale_fill_gradient(
    low = "grey95", 
    high = col_inkjet[1]
  ) +
  theme_bw(base_size = fS7_fs, base_family = ff) +
  theme(
    aspect.ratio = 1,
    axis.title = element_text(size = fS7_fs),
    axis.text = element_text(size = fS7_fs),
    title = element_text(size = fS7_fs),
    strip.text = element_text(size = fS7_fs),
    legend.text = element_text(size = fS7_fs),
    legend.title = element_text(size = fS7_fs),
    legend.key.height = unit(1, "cm")
  ) +
  labs(
    x = "Printer (true)",
    y = "Printer (prediction)", 
    fill = "Count"
  )

inkjet_plot_conf_magenta <- 
  inkjet_cv_preds |>
  filter(
    color == "Magenta", 
    s == 5, 
    pcs == 40, 
    ntrees == 250
  ) |>
  count(printer, pred, name = "n_preds") |>
  ggplot(aes(x = printer, y = pred, fill = n_preds)) +
  geom_tile() +
  geom_label(aes(label = n_preds), size = 6) +
  scale_fill_gradient(
    low = "grey95", 
    high = col_inkjet[2]
  ) +
  theme_bw(base_size = fS7_fs, base_family = ff) +
  theme(
    aspect.ratio = 1,
    axis.title = element_text(size = fS7_fs), 
    axis.text = element_text(size = fS7_fs), 
    title = element_text(size = fS7_fs),
    strip.text = element_text(size = fS7_fs),
    legend.text = element_text(size = fS7_fs),
    legend.title = element_text(size = fS7_fs),
    legend.key.height = unit(1, "cm")
  ) + 
  labs(
    x = "Printer (true)",
    y = "Printer (prediction)",
    fill = "Count"
  )

inkjet_plot_conf_yellow <- 
  inkjet_cv_preds |>
  filter(
    color == "Yellow", 
    s == 20, 
    pcs == 20, 
    ntrees == 1000
  ) |>
  count(printer, pred, name = "n_preds") |>
  ggplot(aes(x = printer, y = pred, fill = n_preds)) +
  geom_tile() +
  geom_label(aes(label = n_preds), size = 6) +
  scale_fill_gradient(
    low = "grey95", 
    high = col_inkjet[3]
  ) +
  theme_bw(base_size = fS7_fs, base_family = ff) +
  theme(
    aspect.ratio = 1,
    axis.title = element_text(size = fS7_fs), 
    axis.text = element_text(size = fS7_fs), 
    title = element_text(size = fS7_fs),
    strip.text = element_text(size = fS7_fs),
    legend.text = element_text(size = fS7_fs),
    legend.title = element_text(size = fS7_fs),
    legend.key.height = unit(1, "cm")
  ) + 
  labs(
    x = "Printer (true)",
    y = "Printer (prediction)",
    fill = "Count"
  )

plot_grid(
  inkjet_plot_conf_cyan,
  inkjet_plot_conf_magenta,
  inkjet_plot_conf_yellow,
  nrow = 2
)


## ----figS8, fig.height = 15, fig.width = 25, out.width = "6.5in", fig.cap = "Principal directions from the best (top row) and worst (bottom row) models for predictions with magenta inkjet signatures. The jfPCs selected are those with the largest PFI values for their respective model. PCs are ordered from left to right based on highest to lowest feature importance."----

# Specify the figure font size
fS8_fs = 28

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
    linesizes = rep(1, 7), 
    linetype = "solid"
  ) +
    scale_x_reverse() +
    theme_bw(base_size = fS8_fs, base_family = ff) +
    theme(
      legend.position = "bottom",
      strip.background = element_rect(color = "white", fill = "white"),
      axis.title = element_text(size = fS8_fs), 
      axis.text = element_text(size = fS8_fs), 
      title = element_text(size = fS8_fs),
      strip.text = element_text(size = fS8_fs),
      legend.text = element_text(size = fS8_fs),
      legend.title = element_text(size = fS8_fs)
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

inkjet_magenta_best_pc_dirs <-
  plot_inkjet_pcs(inkjet_pfi_best[[2]]) + 
  labs(title = "Magenta (best)") + 
  guides(color = guide_legend(override.aes = list(linewidth = 1.5))) + 
  theme(legend.key.width = unit(3.5, "line"))

inkjet_magenta_worst_pc_dirs <-
  plot_inkjet_pcs(inkjet_pfi_worst[[2]]) + 
  labs(title = "Magenta (worst)")
  
plot_grid(
  inkjet_magenta_best_pc_dirs + theme(legend.position = "none"),
  inkjet_magenta_worst_pc_dirs + theme(legend.position = "none"),
  get_plot_component(
    inkjet_magenta_best_pc_dirs,
    'guide-box-bottom', 
    return_all = TRUE
  ),
  ncol = 1,
  rel_heights = c(0.3, 0.3, 0.05)
)



## ----figS9, fig.height = 15, fig.width = 25, out.width = "6.5in", fig.cap = "Principal directions from the best (top row) and worst (bottom row) models for predictions with yellow inkjet signatures. The jfPCs selected are those with the largest PFI values for their respective model. PCs are ordered from left to right based on highest to lowest feature importance."----

inkjet_yellow_best_pc_dirs <-
  plot_inkjet_pcs(inkjet_pfi_best[[3]]) +
  labs(title = "Yellow (best)") +
  guides(color = guide_legend(override.aes = list(linewidth = 1.5))) + 
  theme(legend.key.width = unit(3.5, "line"))

inkjet_yellow_worst_pc_dirs <-
  plot_inkjet_pcs(inkjet_pfi_worst[[3]]) + 
  labs(title = "Yellow (worst)")
  
plot_grid(
  inkjet_yellow_best_pc_dirs + theme(legend.position = "none"),
  inkjet_yellow_worst_pc_dirs + theme(legend.position = "none"),
  get_plot_component(
    inkjet_yellow_best_pc_dirs,
    'guide-box-bottom', 
    return_all = TRUE
  ),
  ncol = 1,
  rel_heights = c(0.3, 0.3, 0.05)
)



## ----figS10, fig.height = 8, fig.width = 20, out.width = "6.5in", fig.cap = "Cross validation average accuracies. Black lines represent CV accuracies from models with PC selected via PFI."----

# Specify the figure font size
fS10_fs = 22

# Plot results in comparison to other accuracies (500 random forest trees):
ggplot() +
  geom_hline(
    data = inkjet_paper_res_best,
    mapping = aes(yintercept = acc_ave),
    linewidth = 1,
    linetype = "dashed"
  ) +
  geom_line(
    data = inkjet_cv_acc_summary |> filter(ntrees == 500),
    aes(
      x = s,
      y = acc_ave,
      group = factor(pcs, levels = seq(10, 100, 10)),
      color = factor(pcs, levels = seq(10, 100, 10))
    ),
    linewidth = 0.75
  ) +
  geom_line(
    data = inkjet_cv_acc_pfi_informed_summary,
    mapping = aes(x = s, y = acc_ave),
    color = "black",
    linewidth = 1.5
  ) +
  facet_grid(. ~ color) +
  scale_color_manual(
    values = wes_palette(
      name = "Zissou1",
      n = 10,
      type = "continuous"
    )
  ) +
  theme_bw(base_size = fS10_fs, base_family = ff) +
  theme(
    strip.background = element_blank(),
    axis.title = element_text(size = fS10_fs),
    axis.text = element_text(size = fS10_fs),
    title = element_text(size = fS10_fs),
    strip.text = element_text(size = fS10_fs),
    legend.text = element_text(size = fS10_fs),
    legend.title = element_text(size = fS10_fs)
  ) +
  guides(color = guide_legend(override.aes = list(linewidth = 1.5))) +
  labs(
    x = "Smoothing Parameter",
    y = "Accuracy",
    color = "Number of PCs"
  )


