# #' Fast NNLS and NNMF
# #'
# #' This package is built for fast non-negative least square (NNLS) regression and non-negative matrix factorization (NNMF).
# #'
# #' @docType package
# #' @name NNLM
# NULL

#' @import Rcpp
#' @importFrom stats runif
#' @importFrom utils tail
#' @useDynLib NNLM, .registration = TRUE
NULL

#' Micro-array data of NSCLC patients
#'
#' This dataset is a random subset (matrix) of micro-array data from a group of Non-Small Cell Lung Caner (NSCLC) patients.
#' It contains 200 probes / genes (row) for 100 patients / samples (column).
#'
#' @name nsclc
#' @docType data
#' @keywords data
NULL
