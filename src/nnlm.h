#define ARMA_NO_DEBUG

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
#include <R.h>

#define NNLS_REL_TOL 1e-6
#define NNMF_REL_TOL 1e-4
#define MAX_ITER 500
#define N_THREADS 1
#define L2_ETA 0
#define L1_BETA 0
#define SHOW_PROGRESS false
#define SHOW_WARNING true
#define TRACE_ERROR false

//using namespace Rcpp;
using namespace arma;

double mse(const mat & A, const mat & W, const mat & H, const mat & W1, const mat & H2);

mat nnls_solver(const mat & H, mat mu, const umat & mask, int max_iter = MAX_ITER, 
	double rel_tol = NNLS_REL_TOL, int n_threads = N_THREADS);

mat nnls_solver_without_missing(mat & WtW, mat & WtA,
	const mat & A, const mat & W, const mat & W1, const mat & H2, const umat & mask,
	const double & eta = L2_ETA, const double & beta = L1_BETA, int max_iter = MAX_ITER, 
	double rel_tol = NNLS_REL_TOL, int n_threads = N_THREADS);

mat nnls_solver_with_missing(const mat & A, const mat & W, const mat & W1, const mat & H2, const umat & mask, 
	const double & eta = L2_ETA, const double & beta = L1_BETA, int max_iter = MAX_ITER, 
	double rel_tol = NNLS_REL_TOL, int n_threads = N_THREADS);

Rcpp::List nnls(const mat & A, const mat & b, int max_iter = MAX_ITER, double tol = NNMF_REL_TOL, 
	int n_threads = N_THREADS, bool show_progress = SHOW_PROGRESS);

Rcpp::List nmf_nnls(const mat & A, int k, double eta = L2_ETA, double beta = L1_BETA, int max_iter = MAX_ITER, 
	double tol = NNMF_REL_TOL, int n_threads = N_THREADS, bool show_progress = SHOW_PROGRESS, bool show_warning = SHOW_WARNING);

Rcpp::List nnmf_generalized(const mat & A, const mat & W1, const mat & H2, mat W, umat Wm, umat Hm,
	int k, double eta = L2_ETA, double beta = L1_BETA, int max_iter = MAX_ITER, double rel_tol = NNMF_REL_TOL, 
	int n_threads = N_THREADS, bool show_progress = SHOW_PROGRESS, bool show_warning = SHOW_WARNING,
	int nnls_max_iter = MAX_ITER, double nnls_rel_tol = NNLS_REL_TOL, int trace = TRACE_ERROR);

Rcpp::List nmf_brunet(const mat & A, int k, int max_iter = MAX_ITER , double rel_tol = NNMF_REL_TOL, 
	int n_threads = N_THREADS, bool show_progress = SHOW_PROGRESS, bool show_warning = SHOW_WARNING);

mat get_H_brunet(const mat & A, const mat & W, int max_iter = MAX_ITER, double rel_tol = NNMF_REL_TOL, 
	int n_threads = N_THREADS, bool show_progress = SHOW_PROGRESS, bool show_warning = SHOW_WARNING);
