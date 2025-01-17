#' Plot principal component directions
#'
#' Function for plotting the functional PC directions
#'
#' @param fpcs Vector of numbers identifying the PCs to include in the plot
#' @param fdasrvf Object output from jointFPCA, horizFPCA, or vertFPCA
#' @param fpca_method Character string specifying the type of elastic fPCA method to use ('jfpca', 'hfpca', or 'vfpca')
#' @param times Optional vector of times (if not included, times will be represented on the interval from 0 to 1)
#' @param digits Number of digits to print in the title for the proportion of variability explained by a PC
#' @param nrow Number of rows to use when creating a grid of plots
#' @param alpha Vector of alpha values associated with lines in plot (length must match number of lines in plot)
#' @param linesizes Vector of line widths associated with lines in plot (length must match number of lines in plot)
#' @param linetype Vector of line types (e.g., "solid" or "dashed") associated with lines in plot (length must match number of lines in plot)
#' @param freey Indicator for whether y-axis should be freed across facets
#'
#' @export plot_pc_directions
#'
#' @importFrom dplyr %>% arrange distinct group_by left_join mutate n pull rename select
#' @importFrom forcats fct_relevel
#' @importFrom ggplot2 .data aes facet_wrap geom_line ggplot labs scale_linetype_manual scale_size_manual theme_bw
#' @importFrom purrr map_df
#' @importFrom stats runif
#' @importFrom stringr str_replace
#' @importFrom tidyr pivot_longer
#'
#' @returns ggplot2 plot of specified principal component directions
#'
#' @examples
#' # Load packages
#' library(dplyr)
#' library(tidyr)
#'
#' # Select a subset of functions from shifted peaks data
#' sub_ids <-
#'   shifted_peaks$data |>
#'   select(data, group, id) |>
#'   distinct() |>
#'   group_by(data, group) |>
#'   slice(1:4) |>
#'   ungroup()
#'
#' # Create a smaller version of shifted data
#' shifted_peaks_sub <-
#'   shifted_peaks$data |>
#'   filter(id %in% sub_ids$id)
#'
#' # Extract times
#' shifted_peaks_times = unique(shifted_peaks_sub$t)
#'
#' # Convert training data to matrix
#' shifted_peaks_train_matrix <-
#'   shifted_peaks_sub |>
#'   filter(data == "Training") |>
#'   select(-t) |>
#'   mutate(index = paste0("t", index)) |>
#'   pivot_wider(names_from = index, values_from = y) |>
#'   select(-data, -id, -group) |>
#'   as.matrix() |>
#'   t()
#'
#' # Obtain veesa pipeline training data
#' veesa_train <-
#'   prep_training_data(
#'     f = shifted_peaks_train_matrix,
#'     time = shifted_peaks_times,
#'     fpca_method = "jfpca"
#'   )
#'
#' # Plot principal directions of PC1
#' plot_pc_directions(
#'   fpcs = 1,
#'   fdasrvf = veesa_train$fpca_res,
#'   fpca_method = "jfpca",
#'   times = -shifted_peaks_times,
#'   linesizes = rep(0.75,5),
#'   alpha = 0.9
#'  )

plot_pc_directions <- function(
    fpcs,
    fdasrvf,
    fpca_method,
    times = NULL,
    digits = 0,
    alpha = 1,
    nrow = 1,
    linesizes = NULL,
    linetype = TRUE,
    freey = FALSE
  ) {

  # Compute prop var
  prop_var = (fdasrvf$latent)^2 / sum((fdasrvf$latent)^2)

  # Get the fPC data
  if (fpca_method %in% c("jfpca", "vfpca")) {
    fpc_df = purrr::map_df(.x = fpcs, .f = function(pc) data.frame(fpc = pc, fdasrvf$f_pca[, , pc]))
    if (is.null(times)) times = fdasrvf$time
  } else if (fpca_method == "hfpca") {
    fpc_df = purrr::map_df(.x = fpcs, .f = function(pc) data.frame(fpc = pc, t(fdasrvf$gam_pca[, , pc])))
    if (is.null(times)) {
      times = seq(0, 1, length.out = dim(fdasrvf$gam_pca)[2])
    } else {
      times = seq(min(times), max(times), length.out = dim(fdasrvf$gam_pca)[2])
    }
  } else {
    stop("'fpca_method' entered incorrectly. Must be 'jfpca', 'vfpca', or 'hfpca'.")
  }

  # Determine the number of standard deviations
  nstds <- (dim(fpc_df)[2] - 2) / 2

  # Create more informative column names
  colnames <- c(paste0("minus", nstds:1, "SD"),
                "Karcher Mean",
                paste0("plus", 1:nstds, "SD"))
  colnames(fpc_df)[-1] = colnames

  # Create names associated with the lines used for plotting
  linenames <- c(paste0("-", nstds:1, "SD"),
                 "Karcher Mean",
                 paste0("+", 1:nstds, "SD"))

  # Adjust principal direction data for plotting
  fpc_df <-
    fpc_df %>%
    dplyr::group_by(.data$fpc) %>%
    dplyr::mutate(index = 1:n(), time = times) %>%
    tidyr::pivot_longer(names_to = "line", cols = -c(.data$fpc, .data$index, .data$time)) %>%
    dplyr::mutate(line = stringr::str_replace(.data$line, "plus", "+")) %>%
    dplyr::mutate(line = stringr::str_replace(.data$line, "minus", "-")) %>%
    dplyr::mutate(line = factor(.data$line, levels = linenames))

  # Specify line sizes and line types
  if (is.null(linesizes)) {
    linesizes = c(seq(0.3, 1, length.out = nstds+1), seq(0.3, 1, length.out = nstds+1)[nstds:1])
  }
  linetpyes = c(rep("dashed", nstds), "solid", rep("dotdash", nstds))

  # Compute fPC percent
  perc_df <-
    data.frame(fpc = fpcs, perc = as.character(round(prop_var[fpcs] * 100, digits))) %>%
    dplyr::mutate(perc = ifelse(as.numeric(.data$perc) < 0.001, "<0.001", .data$perc))

  # Finish preparing the data for the plot
  if (fpca_method == "jfpca") {
    pc_name = "jfPC"
  } else if (fpca_method == "vfpca") {
    pc_name = "vfPC"
  } else if (fpca_method == "hfpca"){
    pc_name = "hfPC"
  }
  plot_df <- fpc_df %>%
    dplyr::left_join(perc_df, by = "fpc") %>%
    dplyr::mutate(fpc_facet = paste0(
      pc_name,
      " ",
      .data$fpc,
      " (",
      .data$perc,
      "%)"
    ))

  fpc_facet_order <-
    plot_df %>%
    dplyr::select(.data$fpc, .data$fpc_facet) %>%
    dplyr::distinct() %>%
    dplyr::pull(.data$fpc_facet)

  # Uncomment for ordering by PC number
  # fpc_facet_order <-
  #   plot_df %>%
  #   dplyr::select(fpc, fpc_facet) %>%
  #   dplyr::distinct() %>%
  #   dplyr::arrange(fpc) %>%
  #   dplyr::pull(fpc_facet)

  plot_df <-
    plot_df %>%
    dplyr::mutate(fpc_facet = factor(.data$fpc_facet, levels = fpc_facet_order))

  # Create plot
  plot <-
    plot_df %>%
    ggplot2::ggplot(ggplot2::aes(
      x = .data$time,
      y = .data$value,
      group = .data$line,
      color = .data$line,
      size = .data$line
    ))
  if (linetype == TRUE) {
    plot <- plot +
      ggplot2::geom_line(aes(linetype =.data$line), alpha = alpha) +
      ggplot2::scale_linetype_manual(values = linetpyes)
  } else {
    plot <- plot + ggplot2::geom_line(alpha = alpha)
  }
  if (freey) {
    plot <-
      plot +
      ggplot2::facet_wrap(. ~ .data$fpc_facet, nrow = nrow, scales = "free_y")
  } else {
    plot <-
      plot +
      ggplot2::facet_wrap(. ~ .data$fpc_facet, nrow = nrow)
  }
  plot +
    ggplot2::theme_bw() +
    ggplot2::scale_size_manual(values = linesizes) +
    ggplot2::labs(color = "", linetype = "", size = "", y = "")

}
