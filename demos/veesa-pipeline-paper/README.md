# Information on Supplementary Files for 'Explainable Machine Learning for Functional Data'

## `code/`

Contains the code associated with manuscript. Run the files in the following order. There are some additional instructions for the scripts in `03-paper-hct-extra/` and `04-paper-inkjet-extra/` included in the body of `02-paper.Rmd`.

- `01-generate-hct-example-data.Rmd`: Contains the code that generates example data for running the H-CT analysis code. (For proprietary reasons, it is not possible to provide the real H-CT data.)
- `02-paper.Rmd`: Contains the code associated with the analysis in the main paper.
- `03-paper-hct-extra/`: Contains additional code associated with the H-CT analyses in the main paper. The VEESA pipeline steps were run using Python scripts to speed up the process.
- `04-paper-inkjet-extra/`: Contains additional code associated with the inkjet analyses in the main paper. The steps in the VEESA pipeline that use elastic shape analysis were run using Python scripts to speed up the process.
- `05-supplement.Rmd`: Contains the code associated with the analyses in the supplemental document.
- `veesa_env.yml`: File for generating the conda environment used to run python code.

## `data/`

Contains the data needed to run the code.

- `hct-clean-example.pkl`: Contains example data for running the H-CT analysis code. (For proprietary reasons, it is not possible to provide the real H-CT data.) This data can also be generated using `01-generate-hct-example-data.Rmd`.
- `RamanInkjet_PrelDataNoBsln1CYANrows.csv`: Raw (cyan) inkjet data. Provided by Patrick Buzzini.
- `RamanInkjet_PrelDataNoBsln2MAGENTArows.csv`: Raw (magenta) inkjet data. Provided by Patrick Buzzini.
- `RamanInkjet_PrelDataNoBsln3YELLOWrows.csv`:  Raw (yellow) inkjet data. Provided by Patrick Buzzini.
- `shifted-peaks.csv`: Contains the 'shifted peaks' data. Note that this data is also available directly from the `veesa` R package. The code never directly reads in this file, but it is included here as a copy.

## `supplement.pdf`

Supplemental document referenced in the main manuscript. Contains results from additional analyses.

## `veesa-0.1.7.zip`

A copy of version 0.1.7 of the `veesa` R package. This code is also available 
on CRAN and GitHub at https://github.com/sandialabs/veesa.