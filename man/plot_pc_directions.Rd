% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/plot_pc_directions.R
\name{plot_pc_directions}
\alias{plot_pc_directions}
\title{Plot principal component directions}
\usage{
plot_pc_directions(
  fpcs,
  fdasrvf,
  fpca_method,
  times = NULL,
  digits = 0,
  alpha = 1,
  nrow = 1,
  linesizes = NULL,
  linetype = TRUE,
  freey = F
)
}
\arguments{
\item{fpcs}{Vector of numbers identifying the PCs to include in the plot}

\item{fdasrvf}{Object output from jointFPCA, horizFPCA, or vertFPCA}

\item{fpca_method}{Character string specifying the type of elastic fPCA method to use ('jfpca', 'hfpca', or 'vfpca')}

\item{times}{Optional vector of times (if not included, times will be represented on the interval from 0 to 1)}

\item{digits}{Number of digits to print in the title for the proportion of variability explained by a PC}

\item{alpha}{XXX}

\item{nrow}{Number of rows to use when creating a grid of plots}

\item{linesizes}{XXX}

\item{linetype}{XXX}

\item{freey}{Indicator for whether y-axis should be freed across facets}
}
\value{
XXX
}
\description{
Function for plotting the functional PC directions
}
