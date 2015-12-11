#include "nnlm.h"

//[Rcpp::export]]
Rcpp::List nnlm(const mat & x, const mat & y, const vec & alpha, const umat & mask, const mat const beta0,
	int max_iter, double rel_tol, int n_threads, int method)
{
	/******************************************************************************************************
	 *                                    Non-Negative Linear Model
	 *                                    -------------------------                 
	 * Description:
	 * 		solve y = x beta
	 * Arguments:
	 * 	x         : design matrix
	 * 	y         : response matrix
	 * 	alpha     : a vector 3, indicating [L2, correlation/anlge, L1] regularization on beta
	 * 	mask      : Mask of beta, s.t. masked entries are no-updated and fixed to inital values
	 * 	beta0     : initial value of beta0. Could be empty
	 * 	max_iter  : Maximum number of iteration
	 * 	rel_tol   : Relative tolerance between two successive iterations, = |e2-e1|/avg(e1, e2)
	 * 	n_threads : Number of threads (openMP)
	 * 	method    : Integer of 1, 2, 3 or 4, which encodes methods
	 * 	          : 1 = sequential coordinate-wise minimization using square loss
	 * 	          : 2 = Lee's multiplicative update with square loss, which is re-scaled gradient descent
	 * 	          : 3 = sequentially quadratic approximated minimization with KL-divergence
	 * 	          : 4 = Lee's multiplicative update with KL-divergence, which is re-scaled gradient descent
	 * Return:
	 * 	A list (Rcpp::List) of
	 * 		coefficient : beta
	 * 		n_iteration : number of iterations
	 * Author:
	 * 	Eric Xihui Lin <xihuil.silence@gmail.com>
	 * Version:
	 * 	2015-12-11
	 ******************************************************************************************************/

	bool any_missing = !A.is_finite();
	mat beta(x.n_cols, y.n_rows);
	if (beta0.empty())
		beta.zeros();
	else
		beta = beta0;
	int nstep;

	if (any_missing)
		nstep = udate_with_missing(beta, x.t(), y, mask, alpha, max_iter, rel_tol, n_threads, method);
	else
		nstep = update(beta, x.t(), y, mask, alpha, max_iter, rel_tol, n_threads, method);

	return Rcpp::List::create(
		Rcpp::Named("coefficient") = beta,
		Rcpp::Named("n_iteration") = nstep
		);
}
