#include "nnlm.h"

//[[Rcpp::export]]
Rcpp::List c_nnlm(const arma::mat & x, const arma::mat & y, const arma::vec & alpha, const arma::umat & mask, const arma::mat & beta0,
	unsigned int max_iter, double rel_tol, int n_threads, int method)
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
	 * 	          : 1 = Sequential coordinate-wise minimization using square loss
	 * 	          : 2 = Lee's multiplicative update with square loss, which is re-scaled gradient descent
	 * 	          : 3 = Sequentially quadratic approximated minimization with KL-divergence
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

	bool any_missing = !y.is_finite();
	mat beta(x.n_cols, y.n_cols);
	if (beta0.empty())
		beta.randu();
	else
		beta = beta0;
	int nstep;

	if (any_missing)
		nstep = update_with_missing(beta, x.t(), y, mask, alpha, max_iter, rel_tol, n_threads, method);
	else
		nstep = update(beta, x.t(), y, mask, alpha, max_iter, rel_tol, n_threads, method);

	return Rcpp::List::create(
		Rcpp::Named("coefficient") = beta,
		Rcpp::Named("n_iteration") = nstep
		);
}
