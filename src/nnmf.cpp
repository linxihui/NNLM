#include "nnlm.h"

//[[Rcpp::export]]
Rcpp::List nnmf(const mat & A, const int k, mat W, mat H, umat Wm, umat Hm,
	const vec & alpha, const vec & beta, const int max_iter, const double rel_tol, 
	const int n_threads, const int trace, const bool show_warning, const int inner_max_iter, 
	const double inner_rel_tol, const int method)
{
	/******************************************************************************************************
	 *              Non-negative Matrix Factorization(NNMF) using alternating scheme
	 *              ----------------------------------------------------------------
	 * Description:
	 * 	Decompose matrix A such that
	 * 		A = W H
	 * Arguments:
	 * 	A              : Matrix to be decomposed
	 * 	W, H           : Initial matrices of W and H, where ncol(W) = nrow(H) = k. # of rows/columns of W/H could be 0
	 * 	Wm, Hm         : Masks of W and H, s.t. masked entries are no-updated and fixed to initial values
	 * 	alpha          : [L2, angle, L1] regularization on W (non-masked entries)
	 * 	beta           : [L2, angle, L1] regularization on H (non-masked entries)
	 * 	max_iter       : Maximum number of iteration
	 * 	rel_tol        : Relative tolerance between two successive iterations, = |e2-e1|/avg(e1, e2)
	 * 	n_threads      : Number of threads (openMP)
	 * 	show_progress  : If to show progress, useful for long computation, suppressed if trace > 0 (on)
	 * 	trace          : # of iterations for each print out. No trace if trace <= 0
	 * 	show_warning   : If to show warning if targeted `tol` is not reached
	 * 	inner_max_iter : Maximum number of iterations passed to each inner W or H matrix updating loop
	 * 	inner_rel_tol  : Relative tolerance passed to inner W or H matrix updating loop, = |e2-e1|/avg(e1, e2)
	 * 	method         : Integer of 1, 2, 3 or 4, which encodes methods
	 * 	               : 1 = sequential coordinate-wise minimization using square loss
	 * 	               : 2 = Lee's multiplicative update with square loss, which is re-scaled gradient descent
	 * 	               : 3 = sequentially quadratic approximated minimization with KL-divergence
	 * 	               : 4 = Lee's multiplicative update with KL-divergence, which is re-scaled gradient descent
	 * Return:
	 * 	A list (Rcpp::List) of 
	 * 		W, H          : resulting W and H matrices
	 * 		mse_error     : a vector of mean square error (divided by number of non-missings)
	 * 		mkl_error     : a vector (length = number of iterations) of mean KL-distance
	 * 		target_error  : a vector of loss (0.5*mse or mkl), plus constraints
	 * 		average_epoch : a vector of average epochs (one complete swap over W and H)
	 * Author:
	 * 	Eric Xihui Lin <xihuil.silence@gmail.com>
	 * Version:
	 * 	2015-12-11
	 ******************************************************************************************************/

	int n = A.n_rows;
	int m = A.n_cols;
	//int k = H.n_rows; // decomposition rank k
	int N_non_missing = n*m;

	vec mse_err(max_iter), mkl_err(max_iter), terr(max_iter), ave_epoch(max_iter);

	// check progression
	bool show_progress = false;
	if (trace == 0) show_progress = true;
	Progress prgrss(max_iter, show_progress);

	double rel_err = rel_tol + 1;
	double terr_last = 1e99;
	double tmp;
	uvec non_missing;
	bool any_missing = !A.is_finite();
	if (any_missing) 
	{
		non_missing = find_finite(A);
		N_non_missing = non_missing.n_elem;
		mkl_err.fill(mean((A.elem(non_missing)+1e-16) % log(A.elem(non_missing)+1e-16) - A.elem(non_missing)));
	}
	else
		mkl_err.fill(mean(mean((A+1e-16) % arma::log(A+1e-16) - A))); // fixed part in KL-dist, mean(A log(A) - A)

	if (Wm.empty())
		Wm.resize(0, n);
	else
		inplace_trans(Wm);
	if (Hm.empty())
		Hm.resize(0, m);

	if (W.empty())
	{
		W.randu(k, n);
		W *= 0.01;
		if (!Wm.empty())
			W.elem(find(Wm > 0)).fill(0.0);
	}
	else
		inplace_trans(W);

	if (H.empty())
	{
		H.randu(k, m);
		H *= 0.01;
		if (!Hm.empty())
			H.elem(find(Hm > 0)).fill(0.0);
	}

	if (trace > 0)
	{
		Rprintf("\n%10s | %10s | %10s | %10s | %10s\n", "Iteration", "MSE", "MKL", "Target", "Rel. Err.");
		Rprintf("--------------------------------------------------------------\n");
	}

	int i = 0;
	for(; i < max_iter && std::abs(rel_err) > rel_tol; i++) 
	{
		Rcpp::checkUserInterrupt();
		prgrss.increment();

		int total_raw_iter = 0;
		if (any_missing)
		{
			// update W
			total_raw_iter = update_with_missing(W, H, A.t(), Wm, alpha, inner_max_iter, inner_rel_tol, n_threads, method);
			// update H
			total_raw_iter += update_with_missing(H, W, A, Hm, beta, inner_max_iter, inner_rel_tol, n_threads, method);

			const mat & Ahat = W.t()*H;
			mse_err(i) = mean(square((A - Ahat).eval().elem(non_missing)));
			mkl_err(i) += mean((-A % trunc_log(Ahat) + Ahat).eval().elem(non_missing));
		}
		else
		{
			// update W
			total_raw_iter = update(W, H, A.t(), Wm, alpha, inner_max_iter, inner_rel_tol, n_threads, method);
			// update H
			total_raw_iter += update(H, W, A, Hm, beta, inner_max_iter, inner_rel_tol, n_threads, method);

			const mat & Ahat = W.t()*H;
			mse_err(i) = mean(mean(square((A - Ahat))));
			mkl_err(i) += mean(mean(-(A+1e-16) % log(Ahat+1e-16) + Ahat));
		}

		ave_epoch(i) = double(total_raw_iter)/(n+m);

		if (method < 3) // mse based
			terr(i) = 0.5*mse_err(i);
		else // KL based
			terr(i) = mkl_err(i);

		// add penalty term back to the loss function (terr)
		if (alpha(0) != alpha(1))
			terr(i) += 0.5*(alpha(0)-alpha(1))*accu(square(W))/N_non_missing;
		if (beta(0) != beta(1))
			terr(i) += 0.5*(beta(0)-beta(1))*accu(square(H))/N_non_missing;
		if (alpha(1) != 0)
			terr(i) += 0.5*alpha(1)*accu(W*W.t())/N_non_missing;
		if (beta(1) != 0)
			terr(i) += 0.5*beta(1)*accu(H*H.t())/N_non_missing;
		if (alpha(2) != 0)
			terr(i) += alpha(2)*accu(W)/N_non_missing;
		if (beta(2) != 0)
			terr(i) += beta(2)*accu(H)/N_non_missing;

		rel_err = (terr_last - terr(i)) / (terr_last + 1e-9);
		terr_last = terr(i);

		if (trace > 0 && ((i+1) % trace == 0 || i == 0))
			Rprintf("%10d | %10.4f | %10.4f | %10.4f | %10.g\n", i+1, mse_err(i), mkl_err(i), terr(i), rel_err);
	}

	if (trace > 0)
	{
		Rprintf("--------------------------------------------------------------\n");
		Rprintf("%10s | %10s | %10s | %10s | %10s\n\n", "Iteration", "MSE", "MKL", "Target", "Rel. Err.");
	}

	if (i >= max_iter)
	{
		if (show_warning && rel_err > rel_tol)
			Rcpp::warning("Target tolerance not reached. Try a larger max.iter.");
	}
	else
	{
		mse_err.resize(i);
		mkl_err.resize(i);
		terr.resize(i);
		ave_epoch.resize(i);
	}

	return Rcpp::List::create(
		Rcpp::Named("W") = W.t(),
		Rcpp::Named("H") = H,
		Rcpp::Named("mse_error") = mse_err,
		Rcpp::Named("mkl_error") = mkl_err,
		Rcpp::Named("target_error") = terr,
		Rcpp::Named("average_epoch") = ave_epoch
		);
}
