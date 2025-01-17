VEESA R Package
================

<!-- badges: start -->

[![R-CMD-check](https://github.com/sandialabs/veesa/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/sandialabs/veesa/actions/workflows/R-CMD-check.yaml)
[![Codecov test
coverage](https://codecov.io/gh/sandialabs/veesa/graph/badge.svg)](https://app.codecov.io/gh/sandialabs/veesa)
<!-- badges: end -->

`veesa` is an R package for implementing the VEESA pipeline for an
explainable approach to training machine learning models with functional
data inputs. See a preprint manuscript describing the approach on
[arXiv](https://arxiv.org/abs/2501.07602). Installing `veesa` can be
implemented using either of the commands below.

``` r
# CRAN
install.packages("veesa")

# Development version from GitHub
remotes::install_github("sandialabs/veesa")
```

Keep reading for an example using `veesa` to implement the VEESA
pipeline.

## Demonstration

#### Set Up and Data Generation

``` r
# Load R packages
library(cowplot)
library(dplyr)
library(ggplot2)
library(purrr)
library(randomForest)
library(tidyr)
library(veesa)

# Specify a color palette
color_pal = wesanderson::wes_palette("Zissou1", 5, type = "continuous")

# Specify colors for PC direction plots
col_plus1 = "#784D8C"
col_plus2 = "#A289AE"
col_minus1 = "#EA9B44"
col_minus2 = "#EBBC88"
col_pcdir_1sd = c(col_plus1, "black", col_minus1)
col_pcdir_2sd = c(col_plus2, col_plus1, "black", col_minus1, col_minus2)
```

Simulate data:

``` r
sim_data = simulate_functions(M = 100, N = 75, seed = 20211130)
```

Separate data into training/testing:

``` r
set.seed(20211130)
id = unique(sim_data$id)
M_test = length(id) * 0.25
id_test = sample(x = id, size = M_test, replace = FALSE)
sim_data = sim_data %>% mutate(data = ifelse(id %in% id_test, "test", "train"))
```

Simulated functions colored by covariates:

![](README_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

Prepare matrices from the data frames:

``` r
prep_matrix <- function(df, train_test) {
  df %>%
    filter(data == train_test) %>%
    select(id, t, y) %>%
    ungroup() %>%
    pivot_wider(id_cols = t,
                names_from = id,
                values_from = y) %>%
    select(-t) %>%
    as.matrix()
}
sim_train_matrix = prep_matrix(df = sim_data, train_test = "train")
sim_test_matrix = prep_matrix(df = sim_data, train_test = "test")
```

Create a vector of times:

``` r
times = sim_data$t %>% unique()
```

#### Alignment and fPCA

Prepare train data

``` r
train_transformed_jfpca <-
  prep_training_data(
    f = sim_train_matrix,
    time = times, 
    fpca_method = "jfpca",
    optim_method = "DPo"
  )
```

Prepare test data:

``` r
test_transformed_jfpca <-
  prep_testing_data(
    f = sim_test_matrix,
    time = times,
    train_prep = train_transformed_jfpca,
    optim_method = "DPo"
  )
```

Plot several PCs:

![](README_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

Compare jfPCA coefficients from train and test data:

<img src="README_files/figure-gfm/unnamed-chunk-11-1.png" width="75%" style="display: block; margin: auto;" />

#### Models

Create response variable:

``` r
x1_train <- 
  sim_data %>% filter(data == "train") %>%
  select(id, x1) %>%
  distinct() %>% 
  pull(x1)
```

Create data frame with PCs and response for random forest:

``` r
rf_jfpca_df <- 
  train_transformed_jfpca$fpca_res$coef %>%
  data.frame() %>%
  rename_all(.funs = function(x) stringr::str_replace(x, "X", "pc")) %>%
  mutate(x1 = x1_train) %>%
  select(x1, everything())
```

Fit random forest:

``` r
set.seed(20211130)
rf_jfpca = randomForest(x1 ~ ., data = rf_jfpca_df)
```

#### PFI

Compute PFI:

``` r
set.seed(20211130)
pfi_jfpca <- compute_pfi(
  x = rf_jfpca_df %>% select(-x1),
  y = rf_jfpca_df$x1,
  f = rf_jfpca,
  K = 10,
  metric = "nmse"
)
```

PFI results (mean of reps):

<img src="README_files/figure-gfm/unnamed-chunk-16-1.png" width="75%" style="display: block; margin: auto;" />

PFI results (variability across reps):

<img src="README_files/figure-gfm/unnamed-chunk-17-1.png" width="75%" style="display: block; margin: auto;" />

Identify the top PC for each elastic fPCA method:

``` r
top_pc_jfpca <- 
  data.frame(pfi = pfi_jfpca$pfi) %>%
  mutate(pc = 1:n()) %>%
  arrange(desc(pfi)) %>%
  slice(1) %>%
  pull(pc)
```

Principal directions of top PC for each jfPCA method:

<img src="README_files/figure-gfm/unnamed-chunk-19-1.png" width="60%" style="display: block; margin: auto;" />
