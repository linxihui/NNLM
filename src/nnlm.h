//disable armadillo matrix/vector boundary checking
#define ARMA_NO_DEBUG

//SUPPORT_OPENMP is defined by R SHLIB
#ifdef SUPPORT_OPENMP
#include <omp.h>
#endif

#include <bits/stdc++.h> 

//[[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

//[[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>

#include <Rcpp.h>
#include <R.h>

#define NNLM_REL_TOL 1e-8
#define NNMF_REL_TOL 1e-6
#define MAX_ITER 500
#define NNMF_INNER_MAX_ITER 10
#define N_THREADS 1
#define TRACE 0
#define SHOW_WARNING true
#define DEFAULT_METHOD 1

//using namespace Rcpp;
using namespace arma;

Rcpp::List nnlm(const mat & x, const mat & y, const vec & alpha, const umat & mat, int max_iter = MAX_ITER,
	double rel_tol = NNLM_REL_TOL, int n_threads = N_THREADS, int method = DEFAULT_METHOD);

Rcpp::List nnmf(const mat & A, mat W, mat H, umat Wm, const umat & Hm, const vec & alpha, const vec & beta,
	int max_iter = MAX_ITER, double rel_tol = NNMF_REL_TOL, int n_threads = N_THREADS, int trace = TRACE,
	bool show_warning = SHOW_WARNING, int inner_max_iter = NNMF_INNER_MAX_ITER,
	double inner_rel_tol = NNLM_REL_TOL, int method = DEFAULT_METHOD);

int update(mat & H, const mat & Wt, const mat & A, const umat & mask, const vec & beta,
	int max_iter = NNMF_INNER_MAX_ITER, double rel_tol = NNLM_REL_TOL,
	int n_threads = N_THREADS, int method = DEFAULT_METHOD);

int update_with_missing(mat & H, const mat & Wt, const mat & A, const umat & mask, const vec & beta,
	int max_iter = NNMF_INNER_MAX_ITER, double rel_tol = NNLM_REL_TOL,
	int n_threads = N_THREADS, int method = DEFAULT_METHOD);

int scd_ls_update(subview_col<double> Hj, const mat & WtW, vec & mu, const subview_col<uword> mask,
	const int & max_iter, const double & rel_tol);

int scd_kl_update(subview_col<double> Hj, const mat & Wt, const vec & Aj, const vec & sumW,
	const subview_col<uword> mask, const vec & beta, const int & max_iter, const double & rel_tol);

int lee_ls_update(subview_col<double> Hj, const mat & WtW, const vec & WtAj, const double & beta3,
	const subview_col<uword> mask, const int & max_iter, const double & rel_tol);

int lee_kl_update(subview_col<double> Hj, const mat & Wt, const vec & Aj, const vec & sumW,
	const subview_col<uword> mask, const vec & beta, const int & max_iter, const double & rel_tol);
