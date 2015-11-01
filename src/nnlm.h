#ifdef SUPPORT_OPENMP
#include <omp.h>
#endif

//[[Rcpp::depends(RcppArmadillo)]]
#include <bits/stdc++.h> 
#include <RcppArmadillo.h>

//[[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>

#include <Rcpp.h>


//using namespace Rcpp;
using namespace arma;

mat nnls_solver(mat H, mat mu, int max_iter, double tol, unsigned int);
RcppExport SEXP c_nnls(SEXP A_, SEXP b_, SEXP max_iter_, SEXP tol_, SEXP n_threads_, SEXP show_progress_);
RcppExport SEXP nmf_nnls(SEXP A_, SEXP k_, SEXP eta_, SEXP beta_, SEXP max_iter_, SEXP tol_, SEXP n_threads_, SEXP show_progress_);

RcppExport SEXP nmf_brunet(SEXP V_, SEXP k_, SEXP max_iter_, SEXP tol_, SEXP n_threads_, SEXP show_progress_);
RcppExport SEXP get_H_brunet(SEXP V_, SEXP W_, SEXP max_iter_, SEXP tol_, SEXP n_threads_, SEXP show_progress_);

// Rcpp::List nmf_ols(mat A, int k, double eta, double beta, int max_iter, double tol); 
