//[[Rcpp::depends(RcppArmadillo)]]
#include <bits/stdc++.h> 
#include <RcppArmadillo.h>
#include <Rcpp.h>

//using namespace Rcpp;
using namespace arma;

mat nnls_solver(mat H, mat mu, int max_iter, double tol);
RcppExport SEXP c_nnls(SEXP A_, SEXP b_, SEXP max_iter_, SEXP tol);
RcppExport SEXP nmf_nnls(SEXP A_, SEXP k_, SEXP eta, SEXP beta, SEXP max_iter_, SEXP tol);

RcppExport SEXP nmf_brunet(SEXP V_, SEXP k_, SEXP max_iter_, SEXP tol);
RcppExport SEXP get_H_brunet(SEXP V_, SEXP W_, SEXP max_iter_, SEXP tol);

// Rcpp::List nmf_ols(mat A, int k, double eta, double beta, int max_iter, double tol); 
