//disable armadillo matrix/vector boundary checking
#define ARMA_NO_DEBUG

#ifdef _OPENMP
#include <omp.h>
#endif

//[[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

//[[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>

#include <Rcpp.h>
#include <R.h>

#define TINY_NUM 1e-16
#define NNLM_REL_TOL 1e-8
#define NNMF_REL_TOL 1e-6
#define MAX_ITER 500
#define NNMF_INNER_MAX_ITER 10
#define N_THREADS 1
#define TRACE_STEP 10
#define SHOW_WARNING true
#define DEFAULT_METHOD 1

//using namespace Rcpp;
using namespace arma;

Rcpp::List c_nnlm(const mat & x, const mat & y, const vec & alpha, const umat & mat, unsigned int max_iter = MAX_ITER,
	double rel_tol = NNLM_REL_TOL, int n_threads = N_THREADS, int method = DEFAULT_METHOD);

Rcpp::List c_nnmf(const mat & A, const unsigned int k, mat W, mat H, umat Wm, const umat & Hm, const vec & alpha, const vec & beta,
	unsigned int max_iter = MAX_ITER, double rel_tol = NNMF_REL_TOL, int n_threads = N_THREADS, const int verbose = 1,
	bool show_warning = SHOW_WARNING, unsigned int inner_max_iter = NNMF_INNER_MAX_ITER,
	double inner_rel_tol = NNLM_REL_TOL, int method = DEFAULT_METHOD, unsigned int trace = TRACE_STEP);

int update(mat & H, const mat & Wt, const mat & A, const umat & mask, const vec & beta,
	unsigned int max_iter = NNMF_INNER_MAX_ITER, double rel_tol = NNLM_REL_TOL,
	int n_threads = N_THREADS, int method = DEFAULT_METHOD);

int update_with_missing(mat & H, const mat & Wt, const mat & A, const umat & mask, const vec & beta,
	unsigned int max_iter = NNMF_INNER_MAX_ITER, double rel_tol = NNLM_REL_TOL,
	int n_threads = N_THREADS, int method = DEFAULT_METHOD);

int scd_ls_update(subview_col<double> Hj, const mat & WtW, vec & mu, const subview_col<uword> mask,
	const unsigned int & max_iter, const double & rel_tol);

int scd_kl_update(subview_col<double> Hj, const mat & Wt, const vec & Aj, const vec & sumW,
	const subview_col<uword> mask, const vec & beta, const unsigned int & max_iter, const double & rel_tol);

int lee_ls_update(subview_col<double> Hj, const mat & WtW, const vec & WtAj, const double & beta3,
	const subview_col<uword> mask, const unsigned int & max_iter, const double & rel_tol);

int lee_kl_update(subview_col<double> Hj, const mat & Wt, const vec & Aj, const vec & sumW,
	const subview_col<uword> mask, const vec & beta, const unsigned int & max_iter, const double & rel_tol);

void add_penalty(const unsigned int & i_e, vec & terr, const mat & W, const mat & H,
	const unsigned int & N_non_missing, const vec & alpha, const vec & beta);
