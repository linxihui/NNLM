#include <cmath>

//SUPPORT_OPENMP is defined by R SHLIB
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


mat nnls_solver(const mat & H, mat mu, int max_iter = 1000, double tol = 1e-4, int n_threads = 0);
Rcpp::List nnls(const mat & A, const mat & b, int max_iter = 1000, double tol = 1e-4, int n_threads = 1, bool show_progress = false);
Rcpp::List nmf_nnls(const mat & A, int k, double eta = 0, double beta = 0, int max_iter = 1000, double tol = 1e-4, 
	int n_threads = 1, bool show_progress = false, bool show_warning = true);
Rcpp::List nmf_partial(const mat & A, const mat & W1, const mat & H2, int k, double eta = 0, double beta = 0, 
	int max_iter = 1000, double tol = 1e-4, int n_threads = 1, bool show_progress = true, bool show_warning = true);

Rcpp::List nmf_brunet(const mat & A, int k, int max_iter = 1000 , double tol = 1e-4, 
	int n_threads = 1, bool show_progress = false, bool show_warning = false);
mat get_H_brunet(const mat & A, const mat & W, int max_iter = 10000, double tol = 1e-4, 
	int n_threads = 1, bool show_progress = false, bool show_warning = false);
// Rcpp::List nmf_ols(mat A, int k, double eta, double beta, int max_iter, double tol); 
